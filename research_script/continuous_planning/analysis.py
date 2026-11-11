"""Question-specific model comparisons, numerical reports and subject figures."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
from models import Engine, make_spec, feature_blocks, compare


def specs(d, actions):
    b = feature_blocks(d, actions)
    s = {}
    definitions = {
        "behavior": ["H"], "current_visual": ["H", "V"], "current_prior": ["H", "V", "P"],
        "past_visual": ["H", "V", "VL"], "baseline": ["H", "V", "VL", "P"],
        "amount": ["H", "V", "VL", "P", "B"],
        "compute": ["H", "V", "VL", "P", "B", "E"],
        "actor": ["H", "V", "VL", "P", "B", "E", "A"],
        "evidence_history": ["H", "V", "VL", "P", "B", "E", "EH"],
        "plan_history": ["H", "V", "VL", "P", "B", "E", "O"],
        "full_history": ["H", "V", "VL", "P", "B", "E", "EH", "O"],
        "past_model_no_visual_history": ["H", "V", "P", "PL"],
        "past_model": ["H", "V", "VL", "P", "PL"],
    }
    for name, keys in definitions.items():
        s[name] = make_spec(name, b, keys, pixels=name != "behavior")
    for lag in range(len(C.LAGS)):
        s[f"no_visual_lag{lag}"] = make_spec(f"no_visual_lag{lag}", b, definitions["baseline"],
            remove=[c for c in b["VL"] if c.startswith(f"lag{lag}_")])
        s[f"no_model_lag{lag}"] = make_spec(f"no_model_lag{lag}", b, definitions["past_model"],
            remove=[c for c in b["PL"] if c.startswith(f"lag{lag}_")])
    pair = ["incumbent_change", "alternative_change", "evidence_available"]
    s["revision_pair"] = make_spec("revision_pair", b, definitions["amount"], extra=pair)
    s["incumbent_only"] = make_spec("incumbent_only", b, definitions["amount"], extra=pair[:1]+pair[2:])
    s["alternative_only"] = make_spec("alternative_only", b, definitions["amount"], extra=pair[1:])
    s["past_before_begin"] = make_spec("past_before_begin", b, definitions["baseline"]+["EH", "O"],
        remove=["mismatch_x_evidence"])
    return s, b


class Reporter:
    def __init__(self, args, module, d, pixels, manifest, existing=None):
        self.args, self.module, self.d = args, module, d
        self.root, self.manifest, self.done = existing or C.start_manifest(args, C.MODULES[module],
            dict(cache_signature=manifest["signature"], cache_manifest=C.sha(C.cache_dir(args)/"manifest.json")))
        self.manifest["smoke"] = manifest["smoke"]
        self.engine = Engine(args, d, pixels, manifest)
        self.specs, self.blocks = specs(d, manifest["num_actions"])
        self.effects, self.runs, self.qa, self.predictions = [], [], {}, {}

    def fit(self, name, outcome="switch", engine=None):
        context = engine.signature if engine is not None else self.engine.signature
        key = (name, outcome, context)
        if key not in self.predictions:
            prediction, qa = (engine or self.engine).fit(self.specs[name], outcome)
            self.predictions[key] = prediction, qa
            self.qa[name+"/"+outcome+"@"+context[:12]] = qa
            # Read-only references avoid writing the same large table once per figure module.
        return self.predictions[key]

    def contrast(self, base, extended, name, outcome="switch", primary="", engine=None):
        bp, bq = self.fit(base, outcome, engine)
        ep, eq = self.fit(extended, outcome, engine)
        summary, by_run, joined = compare(bp, ep, bq, eq, name, self.module, primary)
        summary.update(subject=self.args.subject, game=self.args.game)
        by_run["contrast"] = name
        self.effects.append(summary); self.runs.append(by_run)
        return joined

    def finish(self, extra=None):
        pd.DataFrame(self.effects).to_csv(self.root / "effects.csv", index=False)
        pd.concat(self.runs, ignore_index=True).to_csv(self.root / "run_effects.csv", index=False) if self.runs else pd.DataFrame(columns=["contrast", "run", "delta_ll"]).to_csv(self.root/"run_effects.csv", index=False)
        C.write_json(self.root / "qa.json", dict(models=self.qa, extra=extra or {},
            population_p_values="Computed only by 08_group_report using all six subjects",
            uncertainty="Subject plots show run variation; not population confidence intervals"))
        core = [r for r in self.effects if r.get("primary")]
        C.finish(self.root, self.manifest, contrasts=len(self.effects),
                 primary_estimable=all(r["estimable"] for r in core), **(extra or {}))


def overview_axes(title, reporter=None):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=13)
    if reporter is not None:
        fig.supxlabel(f"sub{reporter.args.subject:03d}, game{reporter.args.game}; N={len(reporter.d):,} recorded decisions; {reporter.d.run.nunique()} source runs. Counts/exclusions: QA tables.", fontsize=8)
    return fig, ax.ravel()


def effect_plot(ax, effects, title="Held-out gain (equal-run mean)"):
    if not effects:
        ax.text(.5, .5, "No estimable comparisons", ha="center"); return
    labels = [e["contrast"] for e in effects]
    values = np.array([e.get("delta_ll", np.nan) for e in effects], float)
    ax.scatter(values, range(len(values)), s=30)
    for i, v in enumerate(values):
        if not np.isfinite(v): ax.text(0, i, "not estimable", fontsize=7)
    ax.set_yticks(range(len(values)), labels, fontsize=7)
    ax.axvline(0, color="gray", lw=1); ax.set_xlabel("Delta log probability (nats)")
    ax.set_title(title, fontsize=10)


def calibration(ax, predictions, title="Out-of-fold calibration"):
    if not len(predictions) or "p_1" not in predictions:
        ax.text(.5, .5, "No valid predictions", ha="center"); return
    q = predictions.assign(bin=pd.cut(predictions.p_1, np.linspace(0, 1, 11), include_lowest=True))
    b = q.groupby("bin", observed=True).agg(p=("p_1", "mean"), y=("target", "mean"), n=("target", "size"))
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.scatter(b.p, b.y, s=np.maximum(10, np.sqrt(b.n)))
    ax.set(xlabel="Predicted probability", ylabel="Observed proportion", title=title)


def residual_surface(ax, data, baseline, x, y, title):
    if not len(baseline):
        ax.text(.5, .5, "No valid baseline", ha="center"); return
    p = baseline[["decision_id", "p_1", "target"]].merge(data[["decision_id", x, y]], on="decision_id")
    p = p.dropna()
    if len(p)<50 or p[x].nunique()<3 or p[y].nunique()<3:
        ax.text(.5, .5, "Insufficient joint support", ha="center"); return
    p["xb"] = pd.qcut(p[x], 5, duplicates="drop")
    p["yb"] = pd.qcut(p[y], 5, duplicates="drop")
    p["residual"] = p.target-p.p_1
    table = p.groupby(["xb", "yb"], observed=True).agg(r=("residual", "mean"), n=("residual", "size"))
    table.loc[table.n < 20, "r"] = np.nan
    grid = table.r.unstack().to_numpy()
    limit = max(.01, np.nanmax(abs(grid))) if np.isfinite(grid).any() else .01
    im = ax.imshow(grid.T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.figure.colorbar(im, ax=ax, label="Observed minus baseline switch probability")
    ax.set(xlabel=x+" (quantile)", ylabel=y+" (quantile)", title=title)
    ax.set_xticks(np.arange(grid.shape[0]), np.arange(1, grid.shape[0]+1))
    ax.set_yticks(np.arange(grid.shape[1]), np.arange(1, grid.shape[1]+1))


def information(r):
    r.contrast("current_visual", "past_visual", "past_visual_increment")
    r.contrast("past_visual", "baseline", "root_increment")
    r.contrast("current_prior", "past_model_no_visual_history", "past_model_before_lagged_visual")
    r.contrast("baseline", "past_model", "past_model_after_lagged_visual")
    for lag in range(len(C.LAGS)):
        r.contrast(f"no_visual_lag{lag}", "baseline", f"visual_lag{lag}")
        r.contrast(f"no_model_lag{lag}", "past_model", f"model_lag{lag}")
    fig, ax = overview_axes("01: Information timescales (real time, not search time)", r)
    ax[0].axis("off")
    ax[0].text(.02, .9, "Observation x[t] -> recorded human action a[t]\n"
        "All imagination events share decision time t.\n"
        "Path action d -> a[t+d-1]; predicted state d -> x[t+d].\n"
        "Mismatch becomes available only when x[t+d] is observed.\n"
        "Lag predictors use only earlier real decisions.", va="top", linespacing=1.8)
    lookup = {e["contrast"]: e["delta_ll"] for e in r.effects}
    mat = np.array([[lookup[f"{kind}_lag{i}"] for i in range(6)] for kind in ("visual", "model")])
    im = ax[1].imshow(mat, aspect="auto", cmap="RdBu_r")
    ax[1].set_yticks([0, 1], ["visual", "model"])
    ax[1].set_xticks(range(6), [f"{lo:g}-{hi:g}" for lo, hi in C.LAGS], rotation=30)
    ax[1].set_xlabel("Past seconds"); fig.colorbar(im, ax=ax[1], label="Drop-band Delta LL")
    effect_plot(ax[2], r.effects[2:4], "Old evaluation before/after visual-history controls")
    for kind in ("visual", "model"):
        ax[3].plot(range(6), [lookup[f"{kind}_lag{i}"] for i in range(6)], "o-", label=kind)
    ax[3].axhline(0, color="gray"); ax[3].legend(); ax[3].set(title="Lag-band support; not a precise reaction delay", xlabel="Lag band", ylabel="Delta LL")
    C.save_figure(fig, r.root, "figure1_information_timescales")


def maintenance(r):
    for outcome in ("switch", "destination"):
        for base, ext, label in [("behavior", "current_visual", "current_environment"),
                                 ("current_visual", "baseline", "history_and_root"),
                                 ("baseline", "amount", "amount"),
                                 ("amount", "compute", "content_given_amount"),
                                 ("compute", "actor", "terminal_actor")]:
            r.contrast(base, ext, label+"_"+outcome, outcome)
        r.contrast("baseline", "compute", "current_compute_"+outcome, outcome,
                   primary="H1" if outcome == "switch" else "H2")
    # The factorized observed-action likelihood uses destination only on switches.
    exact = []
    for name in ("baseline", "amount", "compute", "actor"):
        sw, _ = r.fit(name); dest, _ = r.fit(name, "destination")
        if not len(sw): continue
        merged = sw[["row_id", "run", "target", "logp"]].merge(dest[["row_id", "logp"]], on="row_id", how="left", suffixes=("_switch", "_destination"))
        merged["logp_exact"] = merged.logp_switch+np.where(merged.target==1, merged.logp_destination, 0.)
        summary = merged.groupby("run").logp_exact.mean().reset_index(); summary["model"] = name
        exact.append(summary)
    if exact: pd.concat(exact).to_csv(r.root/"factorized_exact_action.csv", index=False)
    baseline, _ = r.fit("baseline"); full, _ = r.fit("compute")
    # Competing transitions are NON-events in the corresponding one-step cause
    # probability, not discarded observations. Predict destination for stays too.
    cause_scores = {}
    for model_name in ("baseline", "compute"):
        sw, sq = r.fit(model_name)
        _, dq = r.fit(model_name, "destination")
        files = [q.get("destination_risk_file") for q in dq["folds"]]
        if not files or not all(files): continue
        dest_risk = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        risk = dest_risk.merge(sw[["row_id", "p_1"]].rename(columns={"p_1": "p_switch"}), on="row_id", validate="one_to_one")
        out = []
        for cause in ("NOOP_to_active", "active_to_NOOP", "active_to_other_active"):
            q = risk[risk.incumbent==0].copy() if cause=="NOOP_to_active" else risk[risk.incumbent!=0].copy()
            if cause=="NOOP_to_active":
                p = q.p_switch.to_numpy(); y = (q.human_action!=0).to_numpy()
            elif cause=="active_to_NOOP":
                p = (q.p_switch*q.p_0).to_numpy(); y = (q.human_action==0).to_numpy()
            else:
                p = (q.p_switch*(1-q.p_0)).to_numpy(); y = ((q.human_action!=0)&(q.human_action!=q.incumbent)).to_numpy()
            p = np.clip(p, 1e-15, 1-1e-15)
            q["logp_cause"] = y*np.log(p)+(1-y)*np.log1p(-p)
            q["event"] = y.astype(int); q["cause"] = cause
            out.append(q[["row_id", "run", "cause", "logp_cause", "event"]])
        if out: cause_scores[model_name] = pd.concat(out)
    if len(cause_scores)==2:
        joined = cause_scores["baseline"].merge(cause_scores["compute"], on=["row_id", "run", "cause", "event"], suffixes=("_base", "_full"))
        joined["delta_ll"] = joined.logp_cause_full-joined.logp_cause_base
        by_run = joined.groupby(["cause", "run"], as_index=False).agg(delta_ll=("delta_ll", "mean"), rows=("row_id", "size"), events=("event", "sum"))
        by_run.to_csv(r.root/"competing_transition_scores.csv", index=False)
        complete = all(r.fit(name, outcome)[1]["complete"] for name in ("baseline", "compute") for outcome in ("switch", "destination"))
        for cause, rows in by_run.groupby("cause"):
            r.effects.append(dict(subject=r.args.subject, game=r.args.game, contrast=cause, family="02", primary="", outcome="cause_probability",
                delta_ll=rows.delta_ll.mean() if complete else np.nan, estimable=complete,
                reason="" if complete else "Incomplete cause-probability outer folds", rows=int(rows.rows.sum()), runs=len(rows)))
            r.runs.append(rows.assign(contrast=cause))
    # Group-specific comparisons include non-event risk rows, not just event rows.
    if len(baseline) and len(full):
        joined = baseline.merge(full[["row_id", "logp"]], on="row_id", suffixes=("_base", "_full"))
        for label, mask in (("noop_risk", joined.incumbent==0), ("active_risk", joined.incumbent!=0)):
            for run, block in joined[mask].groupby("run"):
                r.runs.append(pd.DataFrame([dict(contrast="risk_"+label, run=run,
                    delta_ll=(block.logp_full-block.logp_base).mean(), rows=len(block))]))
    fig, ax = overview_axes("02: Maintenance, switching and destination", r)
    valid = r.d[r.d.scored & r.d.switch.notna()].copy()
    valid["age_bin"] = pd.cut(valid.bout_age, [0, .125, .25, .5, 1, 2, 4, 8, 16, np.inf], include_lowest=True)
    curves = valid.groupby(["run", "age_bin"], observed=True).switch.mean().reset_index()
    curves.to_csv(r.root/"bout_age_hazard.csv", index=False)
    for run, curve in curves.groupby("run"):
        ax[0].plot(range(len(curve)), curve.switch, alpha=.4, lw=1)
    ax[0].set(title="All-risk bout-age hazard (lines: source runs)", xlabel="Age bin", ylabel="Switch proportion")
    residual_surface(ax[1], r.d, baseline, "root_margin", "evidence_signed", "Baseline-adjusted risk residuals")
    effect_plot(ax[2], [e for e in r.effects if e["contrast"].startswith(("current_compute", "content_given"))])
    calibration(ax[3], full)
    C.save_figure(fig, r.root, "figure2_maintenance_switch_destination")
    # Cause-specific exact probabilities from the coherent two-part model.
    destination, _ = r.fit("compute", "destination")
    if len(destination):
        destination_table = destination.assign(transition=np.where(destination.incumbent==0, "NOOP_to_active",
            np.where(destination.target==0, "active_to_NOOP", "active_to_other_active"))).groupby(
            ["run", "transition", "target"]).agg(logp=("logp", "mean"), count=("target", "size")).reset_index()
        destination_table.to_csv(r.root/"destination_scores.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
        causes = [e for e in r.effects if e["contrast"] in ("NOOP_to_active", "active_to_NOOP", "active_to_other_active")]
        effect_plot(axes[0], causes, "Competing transitions: all at-risk actions included")
        baseline_dest, _ = r.fit("baseline", "destination")
        paired = destination.merge(baseline_dest[["row_id", "logp"]], on="row_id", suffixes=("_full", "_base"))
        paired["gain"] = paired.logp_full-paired.logp_base
        grid = paired.groupby(["incumbent", "target"]).gain.mean().unstack().reindex(index=range(r.engine.n_actions), columns=range(r.engine.n_actions))
        im = axes[1].imshow(grid.to_numpy(), cmap="RdBu_r", aspect="auto")
        fig.colorbar(im, ax=axes[1], label="Conditional destination Delta LL")
        axes[1].set(xlabel="Switch destination", ylabel="Incumbent action", title="Unsupported destinations masked")
        C.save_figure(fig, r.root, "figure2_competing_transitions_destinations")


def revision(r):
    r.contrast("alternative_only", "revision_pair", "incumbent_change_given_alternative")
    r.contrast("incumbent_only", "revision_pair", "alternative_change_given_incumbent")
    r.contrast("compute", "evidence_history", "earlier_evidence_increment")
    r.contrast("compute", "full_history", "all_earlier_computation_increment", primary="H3")
    if "evidence_slope" in r.d:
        from models import Spec
        r.specs["compute_with_slope"] = Spec("compute_with_slope", r.specs["compute"].columns+("evidence_slope",))
        eligible_slope = r.d.copy()
        eligible_slope["scored"] &= np.isfinite(eligible_slope.evidence_slope) & (eligible_slope.evidence_slope_events>=3)
        slope_engine = Engine(r.args, eligible_slope, r.engine.design.pixels,
            dict(signature=r.engine.signature+"-slope-subset", num_actions=r.engine.n_actions))
        r.contrast("compute", "compute_with_slope", "slope_given_endpoint_budget_ge3", engine=slope_engine)
    # Eight-second delayed (non-wrapping) history is a conservative alignment
    # sensitivity. It preserves causal availability, unlike a wrapped future shift.
    past_columns = r.blocks["EH"] + r.blocks["O"]
    shifted = r.d.copy()
    for _, indices in shifted.groupby("episode", sort=False).groups.items():
        ix = np.asarray(indices); time = shifted.loc[ix, "time"].to_numpy()
        source = np.searchsorted(time, time-C.settings(r.args)["time_shift_seconds"], side="right")-1
        values = shifted.loc[ix, past_columns].to_numpy(float)
        replacement = np.full_like(values, np.nan)
        valid = source>=0; replacement[valid] = values[source[valid]]
        shifted.loc[ix, past_columns] = replacement
    delayed_engine = Engine(r.args, shifted, r.engine.design.pixels,
        dict(signature=r.engine.signature+"-causal-delay-8s", num_actions=r.engine.n_actions))
    r.contrast("compute", "full_history", "delayed_history_alignment_sensitivity", engine=delayed_engine)
    # All three predeclared EWMA kernels are jointly regularized, not peak-selected.
    current, _ = r.fit("compute")
    fig, ax = overview_axes("03: Revision and evidence history", r)
    baseline, _ = r.fit("amount")
    residual_surface(ax[0], r.d, baseline, "incumbent_change", "alternative_change", "Separate incumbent/alternative evidence")
    residual_surface(ax[1], r.d, current, "bout_age", "history_cumulative", "Residual beyond current computation")
    effect_plot(ax[2], r.effects)
    episode = r.d.episode.iloc[0]
    sample = r.d[r.d.episode==episode].iloc[:min(450, len(r.d))]
    ax[3].plot(sample.time, sample.evidence_signed, lw=.8, label="current evidence")
    ax[3].plot(sample.time, sample.history_cumulative, lw=.8, label="past bout sum")
    switches = sample[sample.switch==1]
    ax[3].scatter(switches.time, np.zeros(len(switches)), marker="|", color="black", label="human switch")
    ax[3].legend(fontsize=7); ax[3].set(title="First-episode example (descriptive)", xlabel="Seconds")
    C.save_figure(fig, r.root, "figure3_revision_accumulation")


def computation(r, pixels, manifest):
    r.contrast("baseline", "past_before_begin", "past_information_for_initiation", "begin")
    events = pd.read_csv(C.cache_dir(r.args)/"search_events.csv.gz")
    extra = [c for c in r.d if c not in events and c != "decision_id"]
    events = events.merge(r.d[["decision_id"]+extra], on="decision_id", validate="many_to_one")
    C.assert_available(events.origin_time, events.available_time, events.time)
    engine = Engine(r.args, r.d, pixels, manifest, events=events)
    b = r.blocks
    base_cols = ["event_index", "depth_pre", "node_entropy"]
    r.specs["stop_context"] = make_spec("stop_context", b, ["H", "V", "VL", "P"], extra=base_cols)
    r.specs["stop_evidence"] = make_spec("stop_evidence", b, ["H", "V", "VL", "P"],
        extra=base_cols+["prior_evidence_abs", "last_change"])
    r.contrast("stop_context", "stop_evidence", "evidence_for_voluntary_stop", "stop", engine=engine)
    r.contrast("baseline", "amount", "quantity_for_human_switch")
    r.contrast("amount", "compute", "content_given_quantity_for_human_switch")
    curves = events.groupby(["run", "event_index"]).agg(stop=("stop_choice", "sum"),
        risk=("stop_choice", "size"), forced=("forced_terminal", "sum"), resets=("control", lambda x: (x==1).sum())).reset_index()
    curves["hazard"] = curves.stop/curves.risk
    curves.to_csv(r.root/"stop_risk_sets.csv", index=False)
    diagnostics = events.groupby("run").agg(gate_mean=("gate_continue", "mean"),
        actual_continue=("stop_choice", lambda x: 1-x.mean()), voc_mean=("voc_continue_minus_stop", "mean"))
    diagnostics.to_csv(r.root/"gate_diagnostics.csv")
    fig, ax = overview_axes("04: Initiation, continuation and voluntary stopping", r)
    for run, rows in r.d.groupby("run"):
        counts = rows.budget.value_counts(normalize=True).sort_index()
        ax[0].plot(counts.index, counts.values, "o-", alpha=.4, ms=2)
    ax[0].set(yscale="log", xlabel="Additional computations (0 included)", ylabel="Proportion", title="Budget distributions by run")
    for run, rows in curves.groupby("run"):
        ax[1].plot(rows.event_index, rows.hazard, ".-", alpha=.4)
    ax[1].set(title="Voluntary STOP / observed control opportunities", xlabel="Computations already completed", ylabel="STOP hazard")
    initiation, _ = r.fit("past_before_begin", "begin")
    calibration(ax[2], initiation, "Initiation: held-out calibration")
    effect_plot(ax[3], r.effects)
    C.save_figure(fig, r.root, "figure4_dynamic_computation")


def continuity(r):
    r.contrast("compute", "plan_history", "past_path_and_arrived_mismatch")
    no_interaction = list(r.specs["plan_history"].columns)
    no_interaction.remove("mismatch_x_evidence")
    from models import Spec
    r.specs["plan_no_interaction"] = Spec("plan_no_interaction", tuple(no_interaction))
    r.contrast("plan_no_interaction", "plan_history", "mismatch_by_new_evidence")
    forecasts = pd.read_csv(C.cache_dir(r.args)/"forecasts.csv.gz")
    forecasts = forecasts[forecasts.target_id>=0].copy()
    valid_targets = set(r.d.loc[r.d.scored, "decision_id"])
    forecasts = forecasts[forecasts.target_id.isin(valid_targets)]
    reports = []
    for lo, hi in C.HORIZONS:
        for run, block in forecasts[forecasts.depth.between(lo, hi)].groupby("run"):
            weights = 1./block.groupby("target_id").target_id.transform("size")
            for mode in ("all_paths", "prefix_matched"):
                q = block if mode=="all_paths" else block[block.prefix_match]
                if not len(q): continue
                w = weights.loc[q.index]
                def mean(col):
                    mask = q[col].notna()
                    return np.average(q.loc[mask, col], weights=w[mask]) if mask.any() else np.nan
                reports.append(dict(run=run, horizon=f"{lo}-{hi}", mode=mode, n=len(q), targets=q.target_id.nunique(),
                    action_match=mean("action_match"), persistence_match=mean("persistence_match"),
                    root_match=mean("root_match"), pixel_error=mean("pixel_error"),
                    persistence_pixel_error=mean("persistence_pixel_error")))
    table = pd.DataFrame(reports)
    table.to_csv(r.root/"horizon_alignment.csv", index=False)
    r.d.groupby(["run", "carried"]).agg(budget=("budget", "mean"), count=("budget", "size"),
        switch=("switch", "mean")).reset_index().to_csv(r.root/"carry_descriptive.csv", index=False)
    fig, ax = overview_axes("05: Past predictions, realized paths and updating", r)
    ax[0].axis("off")
    examples = forecasts[(forecasts.depth>=2)&forecasts.action_match.notna()].head(3)
    ax[0].text(.01, .95, "Observed trace examples (first eligible paths)\n\n"+"\n\n".join(
        f"origin={row.origin_time:.2f}s path={row.path}\n target={row.target_time:.2f}s matched={row.action_match:g}, prefix={row.prefix_match}"
        for row in examples.itertuples()), va="top", fontsize=8)
    if len(table):
        all_paths = table[table["mode"]=="all_paths"]
        average = all_paths.groupby("horizon", sort=False)[["action_match", "persistence_match", "root_match"]].mean()
        for name in average:
            ax[1].plot(range(len(average)), average[name], "o-", label=name)
        ax[1].set_xticks(range(len(average)), average.index, rotation=30)
        ax[1].legend(fontsize=7); ax[1].set(title="Descriptive agreement; target-weighted", xlabel="Future decision depth")
    baseline, _ = r.fit("compute")
    residual_surface(ax[2], r.d, baseline, "mismatch_pixel", "evidence_signed", "Only arrived, action-prefix-matched errors")
    effect_plot(ax[3], r.effects)
    C.save_figure(fig, r.root, "figure5_continuity_update")


def run(module):
    args = C.parser(C.MODULES[module]).parse_args()
    d, pixels, manifest = C.load_cache(args, visuals=True)
    r = Reporter(args, module, d, pixels, manifest)
    if r.done:
        print(f"Validated complete analysis {r.root}", flush=True); return
    functions = {"01": information, "02": maintenance, "03": revision, "05": continuity}
    if module == "04": computation(r, pixels, manifest)
    else: functions[module](r)
    r.finish()
