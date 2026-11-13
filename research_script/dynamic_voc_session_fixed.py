"""Pooled within-person Analysis 03 with categorical session fixed effects.

One held-out checkpoint per human session. Unpenalized logistic/linear models;
session intercepts; Newey-West covariance on decision-aggregated influence
scores, with zero-filled time gaps and no covariance across runs. Figure 4 and
its out-of-session predictions are preserved, not refitted on pooled data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("voc_native_base", HERE / "03_dynamic_voc_mechanisms.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
VERSION = "dynamic-voc-session-fixed-v2"
LAGS = (64, 128, 256)
PRIMARY_LAG = 128
FIGURES = ("figure1_voc_and_search_continuation", "figure2_voc_and_imagined_returns", "figure3_voc_and_human_transitions")
ID = ["game", "family", "source", "population", "term", "kind"]


def design(frame, predictors, controls):
    """Unchanged focal columns, orthonormal nuisance span including session FE.

    Rank reduction affects only redundant nuisance columns. A focal predictor
    collinear with this span is unidentifiable, rather than rescuing it by
    silently dropping a confound. All scaling is within this pooled fit.
    """
    n = len(frame)
    z = frame[predictors].to_numpy(float)
    scale = z.std(0)
    if np.any(scale < 1e-10):
        raise ValueError("constant_predictor")
    z = (z - z.mean(0)) / scale
    controls = [x for x in dict.fromkeys(controls) if x not in predictors]
    c, names = base.prepare_matrix(frame, controls, np.ones(n, bool)) if controls else (np.empty((n, 0)), [])
    sessions = sorted(frame.session.unique())
    dummy = np.column_stack([frame.session.eq(s).to_numpy(float) for s in sessions[1:]]) if len(sessions) > 1 else np.empty((n, 0))
    nuisance = np.column_stack([np.ones(n), dummy, c])
    eigenvalues, vectors = np.linalg.eigh(nuisance.T @ nuisance / n)
    keep = eigenvalues > max(eigenvalues.max() * 1e-10, 1e-12)
    nuisance = nuisance @ (vectors[:, keep] / np.sqrt(eigenvalues[keep]))
    residual = z - nuisance @ (nuisance.T @ z / n)
    if np.linalg.eigvalsh(residual.T @ residual / n).min() < 1e-8:
        raise ValueError("unidentifiable_predictor")
    return np.column_stack([z, nuisance]), {"session_levels":list(map(int, sessions)),
        "n_session_fixed_effects":len(sessions)-1, "nuisance_rank":int(keep.sum()),
        "nuisance_input_columns":int(len(names)+len(sessions)), "predictor_sd":scale.tolist()}


def temporal_variances(influence, run, step, lags):
    """HAC diagonal from decision-level scores. Multiple imagination choices
    in one real decision form a single score. Excluded decisions remain zero
    in the regular time grid. No FFT wraparound or cross-run products.
    """
    k = influence.shape[1]
    result = {lag:np.zeros(k) for lag in lags}
    units = 0
    for label in pd.unique(run):
        take = np.flatnonzero(run == label)
        indices = step[take].astype(int)
        indices -= indices.min()
        length = int(indices.max()) + 1
        if length > 1000000:
            raise ValueError("Unexpected real-step grid")
        scores = np.zeros((length, k))
        np.add.at(scores, indices, influence[take])
        units += len(np.unique(indices))
        size = 1 << (2 * length - 1).bit_length()
        fft = np.fft.rfft(scores, n=size, axis=0)
        cov = np.fft.irfft(fft.conj() * fft, n=size, axis=0)[:min(max(lags)+1, length)]
        for lag in lags:
            count = min(lag, length-1)
            weights = 1 - np.arange(1, count+1)/(lag+1)
            result[lag] += cov[0] + 2 * (weights[:, None] * cov[1:count+1]).sum(0)
    return {lag:np.maximum(v, 0) for lag, v in result.items()}, units


def fit_pooled(frame, target, predictors, controls, binary, lags=LAGS):
    need = list(dict.fromkeys([target, *predictors, *controls, "session", "run_label", "real_step"]))
    d = frame[need].copy()
    numeric = list(dict.fromkeys([target, *predictors, *controls]))
    d[numeric] = d[numeric].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[target, *predictors]).reset_index(drop=True)
    audit = {"n_rows_available":len(d), "excluded_constant_outcome_sessions":[]}
    if binary and len(d):
        informative = d.groupby("session")[target].nunique()
        excluded = informative.index[informative < 2].tolist()
        audit["excluded_constant_outcome_sessions"] = list(map(int, excluded))
        d = d.loc[~d.session.isin(excluded)].reset_index(drop=True)
    audit.update(n_rows=len(d), n_runs=int(d.run_label.nunique()), n_sessions=int(d.session.nunique()))
    def failure(status):
        return [{"term":p, "hac_lag":lag, "mean":np.nan, "se":np.nan, "ci_low":np.nan,
                 "ci_high":np.nan, "p":np.nan, "status":status, **audit}
                for p in predictors for lag in lags]
    if len(d) < 100 or d[target].nunique() < 2:
        return failure("insufficient")
    y = d[target].to_numpy(float)
    if binary and min((y == 0).sum(), (y == 1).sum()) < 5:
        return failure("rare_outcome")
    try:
        x, diagnostics = design(d, predictors, controls)
    except ValueError as error:
        return failure(str(error))
    audit.update(diagnostics)
    if binary:
        model = LogisticRegression(penalty=None, fit_intercept=False, solver="newton-cholesky", max_iter=150, tol=1e-8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x, y)
        audit["fit_warnings"] = [str(w.message) for w in caught]
        # A separating hyperplane implies that unpenalized logistic MLE has
        # no finite solution, even if the numerical score tolerance passes.
        # Its vanishing residuals can also yield spuriously tiny sandwich SEs.
        signed_margin = (2*y-1) * model.decision_function(x)
        audit["minimum_signed_margin"] = float(signed_margin.min())
        audit["n_events"] = int(y.sum())
        if np.all(signed_margin > 0):
            return failure("complete_separation")
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            return failure("nonconverged")
        coefficients = model.coef_[0]
        mu = model.predict_proba(x)[:, 1]
        hessian = x.T @ (x * (mu * (1-mu))[:, None])
    else:
        y = (y-y.mean())/y.std()
        hessian = x.T @ x
        coefficients = np.linalg.solve(hessian, x.T @ y)
        mu = x @ coefficients
    residual = y-mu
    audit["gradient_max_per_row"] = float(np.abs(x.T @ residual).max()/len(d))
    audit["n_parameters"] = x.shape[1]
    if binary and audit["gradient_max_per_row"] > 1e-6:
        return failure("nonconverged_score")
    if np.linalg.eigvalsh(hessian).min() <= 1e-10:
        return failure("singular_information")
    k = len(predictors)
    bread = np.linalg.solve(hessian, np.eye(x.shape[1])[:, :k])
    influence = (x @ bread) * residual[:, None]
    variances, n_decisions = temporal_variances(influence, d.run_label.to_numpy(), d.real_step.to_numpy(), lags)
    audit["n_decisions"] = n_decisions
    rows = []
    correction = len(d)/(len(d)-x.shape[1])
    for lag in lags:
        for j, term in enumerate(predictors):
            se = float(np.sqrt(variances[lag][j]*correction))
            beta = float(coefficients[j])
            p = float(2*stats.norm.sf(abs(beta/se))) if se > 0 else np.nan
            rows.append({"term":term, "hac_lag":lag, "mean":beta, "se":se, "ci_low":beta-1.95996398454*se,
                         "ci_high":beta+1.95996398454*se, "p":p, "status":"ok", **audit})
    return rows


def correct(df, keys):
    df = df.copy(); df["q"] = np.nan
    for _, ix in df.groupby(keys, dropna=False).groups.items():
        df.loc[ix, "q"] = base.bh(df.loc[ix, "p"])
    return df


def pooled_associations(d, e):
    records = []
    subject, game = int(d.subject.iloc[0]), int(d.game.iloc[0])
    event_controls = ["margin", "root_value", "root_entropy", "carried", "depth", "log_search_steps"] + base.action_controls(d)
    def fit(part, family, source, population, target, predictors, controls, binary, rename=None):
        result = fit_pooled(part, target, predictors, controls, binary)
        for row in result:
            if rename: row["term"] = rename
            row.update(subject=subject, game=game, family=family, source=source, population=population,
                       kind="log_odds" if binary else "standardized_slope", estimand="held_out", unit="pooled_decisions")
            records.append(row)
    voluntary = e.loc[e.voluntary & e.scored.eq(1)]
    for source in base.SOURCES:
        voc = f"voc_{source}"
        for pop, part in (("all_choices", voluntary), ("first_choice", voluntary.loc[voluntary.choice_index.eq(0)]),
                          ("after_computation", voluntary.loc[voluntary.search_steps_pre.gt(0)])):
            fit(part, "Q1_gate", source, pop, "continue", [voc], event_controls+["return_gain"], True)
        allocation = d.loc[d.scored.eq(1)].copy()
        for outcome in ("additional_computations", "max_depth"):
            allocation["log_allocation"] = np.log1p(allocation[outcome])
            fit(allocation, "Q1_allocation", source, "all_decisions", "log_allocation", [f"{source}_initial"],
                base.base_columns(d), False, rename=outcome)
        searched = voluntary.loc[voluntary.search_steps_pre.gt(0)]
        for metric in base.RETURN_METRICS:
            fit(searched, "Q2_return", source, "after_computation", voc, [metric], event_controls, False)
            for depth in (1, 2, 3):
                fit(searched.loc[searched.depth.eq(depth)], "Q2_fixed_depth", source, f"depth_{depth}", voc, [metric], event_controls, False)
        updates = e.loc[e.has_next_choice & e.search_computation.eq(1) & e.scored.eq(1)]
        fit(updates, "Q2_update", source, "executed_computations", f"delta_voc_{source}", ["return_update"], event_controls+[voc, "bootstrap_return"], False)
        for population in base.POPULATIONS:
            part = base.risk_set(d, population)
            part = part.loc[part.stop_choice_censored.eq(0) & part.forced_stop.eq(0)]
            fit(part, "Q3_human", source, population, "outcome", [f"{source}_initial", f"{source}_change"],
                base.base_columns(d)+base.SEARCH_CONTROL, True)
        base.log(f"sub{subject:03d} {base.GAMES[game]}: {source} pooled session-FE regressions complete")
    return correct(pd.DataFrame(records), ["subject", "game", "family", "source", "kind", "hac_lag"])


def group_summary(effects):
    records = []
    data = effects.loc[effects.hac_lag.eq(PRIMARY_LAG)]
    for key, part in data.groupby(ID, dropna=False):
        values = part.loc[part.status.eq("ok"), "mean"].dropna().to_numpy()
        n = len(values); mean = values.mean() if n else np.nan
        ci = stats.t.ppf(.975, n-1)*stats.sem(values) if n > 1 else np.nan
        records.append(dict(zip(ID, key)) | {"mean":mean, "ci_low":mean-ci, "ci_high":mean+ci,
            "p":base.exact_sign_flip(values), "n_subjects":n, "n_positive":int((values > 0).sum()),
            "n_negative":int((values < 0).sum()), "status":"ok" if n >= 2 else "insufficient",
            "hac_lag":PRIMARY_LAG, "estimand":"held_out", "unit":"subject"})
    return correct(pd.DataFrame(records), ["game", "family", "source", "kind"])


def draw_effects(ax, frame, terms, labels, group=False):
    for i, (term, label) in enumerate(zip(terms, labels)):
        part = frame.loc[frame.term.eq(term)]
        if part.empty or not np.isfinite(part.iloc[0]["mean"]):
            ax.text(.02, i+.15, label+": not estimable", transform=ax.get_yaxis_transform(), fontsize=9)
            continue
        r = part.iloc[0]; beta = r["mean"]
        color = base.COLORS[i % 4]
        ax.errorbar(beta, i, xerr=[[max(0,beta-r.ci_low)], [max(0,r.ci_high-beta)]], fmt="o", color=color, capsize=3)
        q = "<.001" if r.q < .001 else f"{r.q:.3f}"
        count = f"n={int(r.n_subjects)} subjects" if group else f"{int(r.n_decisions):,} decisions"
        ax.text(.02, i+.2, f"{label}   q={q}; {count}", transform=ax.get_yaxis_transform(), fontsize=8.5)
    ax.axvline(0, color=".55", ls="--", lw=.8)
    ax.set_yticks([]); ax.set_ylim(-.5, len(terms)+.15)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    formatter = ScalarFormatter(useMathText=True); formatter.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter)


def render(fig_dir, effects, curves, game, subject=None):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    group = subject is None
    unit = "subject" if group else "run"
    c = base.unit_curves(curves, unit)
    s = effects.loc[effects.source.eq("ema") & effects.hac_lag.eq(PRIMARY_LAG)]
    who = "Group" if group else f"S{subject}"
    note = "Coefficients: equal-subject mean ± 95% t interval; q: subject sign-flip + FDR." if group else "Coefficients: pooled session-FE fit ± 95% HAC interval (128 decisions); q: within-subject FDR."
    def save(fig, name):
        fig.suptitle(f"{base.GAMES[game]} · {who} · pooled regression with session fixed effects", fontsize=14)
        fig.text(.015, .026, note+" EMA VoC; held-out replay sessions.", fontsize=8)
        fig.text(.015, .009, f"Curves are descriptive: thin lines = {unit}s; thick lines = equal-{unit} means. Curves are not adjusted regression predictions.", fontsize=8)
        fig.tight_layout(rect=(0,.065,1,.95), w_pad=2.5, h_pad=3)
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir/f"{name}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for i, pop in enumerate(("first_choice", "after_computation")):
        for outcome, label, color in (("continue", "Executed CONTINUE", base.COLORS[0]), ("voc_soft_continue_probability", "Soft gate probability", base.COLORS[1])):
            part = c.loc[c.family.eq("Q1_gate") & c.source.eq("ema") & c.population.eq(pop) & c.outcome.eq(outcome)]
            base.draw_curve(axes[0,i], part, label, color)
        axes[0,i].set(xlabel="VoC before choice", ylabel="Probability", title="First choice" if i == 0 else "After at least one computation")
        axes[0,i].legend(fontsize=8)
        draw_effects(axes[1,i], s.loc[s.family.eq("Q1_gate") & s.population.eq(pop)], ["voc_ema"], ["VoC → CONTINUE"], group)
        axes[1,i].set(title="Session-adjusted continuation", xlabel="Log-odds per SD of VoC")
    for metric, label, color in (("additional_computations", "Computation count", base.COLORS[0]), ("max_depth", "Maximum depth", base.COLORS[1])):
        base.draw_curve(axes[0,2], c.loc[c.family.eq("Q1_allocation") & c.source.eq("ema") & c.outcome.eq(metric)], label, color)
    axes[0,2].set(xlabel="Initial VoC", ylabel="Count / depth", title="Observed search allocation")
    axes[0,2].legend(fontsize=8)
    draw_effects(axes[1,2], s.loc[s.family.eq("Q1_allocation")], ["additional_computations", "max_depth"], ["Computation count", "Maximum depth"], group)
    axes[1,2].set(title="Session-adjusted allocation", xlabel="Standardized slope: log(1 + allocation)")
    save(fig, FIGURES[0])
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, metric, label in zip(axes.flat, base.RETURN_METRICS, base.RETURN_LABELS):
        base.draw_curve(ax, c.loc[c.family.eq("Q2_return") & c.source.eq("ema") & c.x_name.eq(metric)], "Available before choice", base.COLORS[0])
        ax.set(xlabel=label, ylabel="VoC", title="After at least one computation")
    draw_effects(axes[1,1], s.loc[s.family.eq("Q2_return")], list(base.RETURN_METRICS), ["Reward sum", "Discounted reward", "Bootstrapped return", "Gain over root value"], group)
    axes[1,1].set(title="Session-adjusted return–VoC", xlabel="Standardized slope")
    draw_effects(axes[1,2], s.loc[s.family.eq("Q2_update")], ["return_update"], ["Return change → next VoC change"], group)
    axes[1,2].set(title="Session-adjusted successive updates", xlabel="Standardized slope")
    save(fig, FIGURES[1])
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5))
    for i, (pop, label) in enumerate(zip(base.POPULATIONS, base.POP_LABELS)):
        base.draw_curve(axes[0,i], c.loc[c.family.eq("Q3_human") & c.source.eq("ema") & c.population.eq(pop)], "Descriptive", base.COLORS[0])
        axes[0,i].set(xlabel="Initial VoC", ylabel="Human transition probability", title=label)
        draw_effects(axes[1,i], s.loc[s.family.eq("Q3_human") & s.population.eq(pop)], ["ema_initial", "ema_change"], ["Initial VoC", "Terminal − initial VoC"], group)
        axes[1,i].set(title="With session and task controls", xlabel="Log-odds per SD")
    save(fig, FIGURES[2])


def load_pooled(args, subject, game):
    decisions, events, qa, provenance = [], [], [], []
    for cp, path, config, binding in base.discover(args.shared_cache_root, subject, game, args.checkpoints):
        base.log(f"sub{subject:03d} {base.GAMES[game]}: reading checkpoint {cp}")
        d, e, audit = base.load_cache(path, cp, config, subject, game)
        decisions.append(d.loc[d.held_out].copy()); events.append(e.loc[e.held_out].copy())
        qa.append(audit)
        provenance.append({"checkpoint":cp, "path":str(path), "binding":binding,
                           "manifest_sha256":base.sha256(path/"manifest.json"), "binding_sha256":base.sha256(path/"checkpoint_binding.json"),
                           "configuration":{key:config.get(key) for key in ("discounting", "max_search_steps", "max_depth", "tree_carry", "voc_gate_q_temperature", "voc_eval_stochastic")}})
    d, e = pd.concat(decisions, ignore_index=True), pd.concat(events, ignore_index=True)
    base.validate_keys(d, "pooled held-out decisions")
    base.validate_keys(e, "pooled held-out events", ["event_order"])
    if not d.session.eq(d.checkpoint).all() or not e.session.eq(e.checkpoint).all():
        raise ValueError("Included an imitation-training session")
    if set(d.session.unique()) != set(args.checkpoints):
        raise ValueError("Incomplete pooled session set")
    return d, e, qa, provenance


def old_table(folder, current, legacy):
    path = folder/"tables"/current
    return pd.read_csv(path if path.exists() else folder/"tables"/legacy)


def subject_job(task):
    args, subject, game = task
    hashes = {p.name:base.sha256(p) for p in (Path(__file__), HERE/"03_dynamic_voc_mechanisms.py")}
    d, e, qa, provenance = load_pooled(args, subject, game)
    base.log(f"sub{subject:03d} {base.GAMES[game]}: pooled {len(d):,} unique decisions across {d.session.nunique()} sessions")
    effects = pooled_associations(d, e)
    source = args.output_root/f"sub{subject:03d}_game{game}"
    curves = old_table(source, "curves_heldout.csv", "curves_by_checkpoint.csv")
    curves = curves.loc[curves.held_out.eq(True)]
    prediction = old_table(source, "prediction_inference.csv", "run_inference.csv")
    prediction = prediction.loc[prediction.family.eq("Q4_prediction")]
    out = args.stage/f"sub{subject:03d}_game{game}"; (out/"tables").mkdir(parents=True, exist_ok=True)
    effects.to_csv(out/"tables/pooled_session_fixed_effects.csv", index=False)
    curves.to_csv(out/"tables/curves_heldout.csv", index=False)
    prediction.to_csv(out/"tables/prediction_inference.csv", index=False)
    render(out/"figures", effects, curves, game, subject)
    if any(base.sha256(HERE/name) != digest for name, digest in hashes.items()):
        raise RuntimeError("Source changed while analysis was running")
    base.write_json(out/"analysis_manifest.json", {"status":"COMPLETED", "version":VERSION, "subject":subject,
        "game":game, "source_hashes":hashes, "source_caches":provenance, "qa":qa,
        "pooled_decisions":len(d), "pooled_control_choices":len(e), "n_runs":int(d.run_label.nunique()),
        "sessions":sorted(map(int,d.session.unique())), "primary_hac_lag":PRIMARY_LAG,
        "hac_sensitivity_lags":list(LAGS), "formula":"target ~ focal_predictors + controls + C(session)",
        "subject_inference":"decision-aggregated within-run Newey-West sandwich; asymptotic normal Wald; family FDR",
        "prediction":"Figure 4 and predictions preserved from session-held-out readout; not pooled or refitted"})
    base.log(f"Completed session-FE sub{subject:03d} game{game}")


def aggregate(args, subjects, game):
    effects, curves = [], []
    for subject in subjects:
        root = args.stage/f"sub{subject:03d}_game{game}"
        m = json.loads((root/"analysis_manifest.json").read_text())
        if m["version"] != VERSION or m["source_hashes"][Path(__file__).name] != base.sha256(__file__):
            raise ValueError("Stale subject estimates")
        effects.append(pd.read_csv(root/"tables/pooled_session_fixed_effects.csv"))
        curves.append(pd.read_csv(root/"tables/curves_heldout.csv"))
    e, c = pd.concat(effects, ignore_index=True), pd.concat(curves, ignore_index=True)
    summary = group_summary(e)
    out = args.stage/f"group_game{game}"; (out/"tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(out/"tables/group_inference.csv", index=False)
    e.to_csv(out/"tables/pooled_subject_effects.csv", index=False)
    c.to_csv(out/"tables/curves_heldout.csv", index=False)
    prediction = old_table(args.output_root/f"group_game{game}", "prediction_inference.csv", "group_inference.csv")
    prediction.loc[prediction.family.eq("Q4_prediction")].to_csv(out/"tables/prediction_inference.csv", index=False)
    render(out/"figures", summary, c, game)
    base.write_json(out/"analysis_manifest.json", {"status":"COMPLETED", "version":VERSION, "game":game,
        "subjects":subjects, "inference":"equal-subject mean of pooled FE coefficients; exact subject sign-flip and FDR",
        "source_sha256":base.sha256(__file__), "prediction":"Figure 4 unchanged"})


def documentation(args):
    text = """# Analysis 03: pooled regression with session fixed effects

Figures 1–3 now use participant-by-game regression across all held-out human sessions:
`outcome ~ focal predictors + original controls + C(session)`.
Each session contributes only the checkpoint that excluded it from imitation training.
No repeated checkpoint versions of a human decision enter the pooled regression.

- Binary outcomes: unpenalized logistic regression; continuous outcomes: linear regression.
- Focal predictors are standardized across the pooled fit; continuous outcomes are also standardized.
- Session is categorical (separate intercepts), not a continuous session-number covariate.
- Redundant nuisance columns are reduced without dropping identifiable focal predictors.
  Focal variables explained completely by nuisance variables are marked unidentifiable.
- Binary sessions with no outcome variation carry no finite-intercept within-session
  slope information and are excluded with counts recorded. Unestimable fits are not zeros.
- Within-person inference: Newey-West sandwich covariance on influence scores summed
  within each real decision, then placed on that run's real-step time grid. Missing/risk-set
  decisions are zero-filled, and covariance never crosses runs. Bartlett bandwidth 128
  decisions is primary; 64 and 256 are saved sensitivity analyses. This handles local serial
  dependence and multiple imagined steps within a human decision; it is not a bootstrap
  or protection against arbitrary session-wide dependence/nonstationarity.
- Confidence intervals are asymptotic normal 95% intervals. BH-FDR is separately within
  participant/game/question/source/outcome-type/bandwidth. Primary source is EMA VoC.
- Group inference still uses subjects, not decisions: equal-subject mean FE coefficients,
  95% t interval, exact subject sign-flip p, and family FDR.

Figure 1: VoC and voluntary CONTINUE; initial VoC and log(1+computation count/depth).
Allocation now has adjusted standardized regression slopes, not the old Spearman correlation.
Figure 2: imagined reward sum, discounted reward, bootstrapped return, gain over root;
return change versus the strictly next VoC change. Depth 1/2/3 sensitivities in CSV.
Figure 3: human switching overall, NOOP→active, active→NOOP, active→other active.
Figure 4 is unchanged: independent-session predictive gains from the original readout.
Its labels and q values use run-based inference, not pooled session-FE coefficient inference.

Upper curves remain descriptive equal-run/equal-subject summaries, not adjusted fitted
probabilities. Lower panels show pooled FE coefficients and the stated intervals/tests.
Native cap/forced-STOP handling, action counts, temporal ordering and source-cache checks
are inherited from the validated native-cache loader. Imagined returns are not actual utility.

Tables: `pooled_session_fixed_effects.csv` (all three HAC bandwidths),
`group_inference.csv` (group FE effects), `prediction_inference.csv` (unchanged Q4).
Old figures, tables and reports are archived before installation. Original caches and
the saved Q4 prediction values are unchanged. Archive files retain their original methods.

Rerun all (CPU, no training):
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research_script/03_dynamic_voc_mechanisms.py --subjects 1 2 3 4 5 6 --games 0 1 2 --workers 3
Use --render-only to redraw a completed staged analysis without loading caches or fitting.
"""
    (args.stage/"README.md").write_text(text)
    (args.stage/"source").mkdir(exist_ok=True)
    for path in (Path(__file__), HERE/"03_dynamic_voc_mechanisms.py", HERE/"03_actor_integration_mechanisms.py", HERE/"tests/test_dynamic_voc_session_fixed.py"):
        shutil.copy2(path, args.stage/"source"/path.name)


def install(args, subjects, games):
    # The entire requested analysis finishes in staging before canonical figures
    # are replaced. Archive every overwritten artifact, and retire stale reports.
    archive = args.output_root/"archive"/("before_session_fixed_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder_names = [f"sub{s:03d}_game{g}" for g in games for s in subjects]
    if len(subjects) > 1: folder_names += [f"group_game{g}" for g in games]
    protected = [p for name in folder_names for p in (args.output_root/name).rglob("*")
                 if p.is_file() and ("figure4_" in p.name or p.name.startswith("heldout_"))]
    before = {str(p.relative_to(args.output_root)):base.sha256(p) for p in protected}
    stage_files = [p for p in args.stage.rglob("*") if p.is_file()]
    targets = {p.relative_to(args.stage) for p in stage_files}
    stale = [Path("RESULTS.md"), Path("validation.json"), Path("analysis_run.log")]
    for name in folder_names:
        stale += [Path(name)/"tables"/f for f in ("run_effects_by_checkpoint.csv", "curves_by_checkpoint.csv", "run_inference.csv")]
    for relative in sorted(targets | set(stale)):
        old = args.output_root/relative
        if old.is_file():
            dest = archive/relative; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, dest)
            if relative not in targets: old.unlink()
    for name in ("within_subject_audit",):
        old = args.output_root/name
        if old.exists():
            dest = archive/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), dest)
    for path in stage_files:
        dest = args.output_root/path.relative_to(args.stage)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_name(dest.name+".tmp")
        shutil.copy2(path, temporary); temporary.replace(dest)
    after = {name:base.sha256(args.output_root/name) for name in before}
    if before != after:
        raise RuntimeError("Protected Figure 4/prediction artifacts changed")
    base.write_json(args.output_root/"session_fixed_installation.json", {"version":VERSION, "archive":str(archive),
        "subjects":subjects, "games":games, "installed_files":len(stage_files), "protected_figure4_and_prediction_hashes":before,
        "protected_files_unchanged":True, "source_sha256":base.sha256(__file__)})
    base.log(f"Installed pooled session-FE figures; previous files archived at {archive}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared-cache-root", type=Path, default=base.DEFAULT_CACHE)
    p.add_argument("--output-root", type=Path, default=base.DEFAULT_OUTPUT)
    p.add_argument("--subject", type=int); p.add_argument("--subjects", type=int, nargs="+")
    p.add_argument("--game", type=int, choices=base.GAMES); p.add_argument("--games", type=int, nargs="+", choices=base.GAMES)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1,2,3,4])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--stage-only", action="store_true")
    p.add_argument("--render-only", action="store_true")
    args = p.parse_args()
    subjects = args.subjects or ([args.subject] if args.subject else [])
    games = args.games or ([args.game] if args.game is not None else [])
    if not subjects or not games or len(set(subjects)) != len(subjects) or args.workers < 1:
        p.error("Provide unique subjects, games, and positive workers")
    args.stage = args.output_root/"_session_fixed_build"
    args.stage.mkdir(parents=True, exist_ok=True)
    documentation(args)
    tasks = [(args,s,g) for g in games for s in subjects]
    if args.render_only:
        for _, s, g in tasks:
            root = args.stage/f"sub{s:03d}_game{g}"
            render(root/"figures", pd.read_csv(root/"tables/pooled_session_fixed_effects.csv"), pd.read_csv(root/"tables/curves_heldout.csv"), g, s)
    elif args.workers == 1:
        for task in tasks: subject_job(task)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(subject_job, tasks))
    if len(subjects) > 1:
        for g in games: aggregate(args, subjects, g)
    if not args.stage_only: install(args, subjects, games)


if __name__ == "__main__":
    main()
