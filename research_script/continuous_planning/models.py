"""Session-contained nested prediction and fixed-action-space conditional choice.

Every model, including preprocessing and inner selection, is fit without the
outer block. No second-stage fit mixes the leave-session-out checkpoints.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer

import common as C


def feature_blocks(d, num_actions):
    history = ["bout_age", "bout_left_censored", "previous_reward", "run_progress"]
    history += [f"previous_action_{i}" for i in range(1, 5)]
    history += [c for c in d if c.startswith(("action_fraction_", "switch_count_"))]
    visual = [f"visual_{i}" for i in range(40)]
    visual_past = [c for c in d if c.startswith("lag") and ("visual_" in c or c.endswith("coverage"))]
    prior = ["root_value", "root_entropy", "root_margin", "carried", "useful_carry", "carried_visits", "carried_expanded"]
    prior += [f"root_{s}_{a}" for s in ("p", "q", "n") for a in range(num_actions)]
    amount = ["budget", "begin", "expansion", "reset", "revisit", "max_depth", "branches"]
    evidence = [f"{key}_{a}" for key in ("evaluated", "revision", "first_return") for a in range(num_actions)]
    evidence += ["incumbent_change", "alternative_change", "evidence_abs", "evidence_available", "evidence_signed"]
    past = [c for c in d if c.startswith("history_") and not c.endswith("time")]
    plans = [c for c in d if c.startswith("old_plan_support_")]
    plans += ["old_plan_count", "mismatch_pixel", "mismatch_descriptor", "mismatch_persistence", "mismatch_count", "mismatch_x_evidence"]
    actor = [f"actor_p_{a}" for a in range(num_actions)]
    old_model = [c for c in d if c.startswith("lag") and any(s in c for s in ("root_margin", "root_value", "evidence", "budget"))]
    return dict(H=history, V=visual, VL=visual_past, P=prior, B=amount, E=evidence,
                EH=past, O=plans, A=actor, PL=old_model)


@dataclass(frozen=True)
class Spec:
    name: str
    columns: tuple
    pixels: bool = True


def make_spec(name, blocks, keys, pixels=True, extra=(), remove=()):
    columns = list(dict.fromkeys([c for k in keys for c in blocks[k]] + list(extra)))
    columns = tuple(c for c in columns if c not in set(remove))
    forbidden = {"human_action", "switch", "terminal", "target_action", "action_match", "forced_stop", "voluntary_stop"}
    if forbidden.intersection(columns):
        raise ValueError("Outcome/post-control field requested as predictor")
    return Spec(name, columns, pixels)


def check_control_features(spec, outcome):
    if outcome not in ("begin", "stop"):
        return
    future = {"budget", "begin", "expansion", "reset", "revisit", "max_depth", "branches",
              "incumbent_change", "alternative_change", "evidence_abs", "evidence_signed",
              "evidence_available", "mismatch_x_evidence"}
    if future.intersection(spec.columns) or any(c.startswith(("actor_p_", "revision_", "first_return_", "evaluated_", "root_q_after_", "root_n_after_", "evidence_slope")) for c in spec.columns):
        raise ValueError("Post-control/terminal information requested in initiation or stopping design")


def outer_folds(d):
    for session, rows in d.groupby("session", sort=True):
        if rows.checkpoint.nunique() != 1:
            raise ValueError("Session contains multiple checkpoint feature generators")
        runs = sorted(rows.run.unique())
        if len(runs) < 2:
            raise ValueError("Need >=2 source blocks in each session")
        for run in runs:
            train = rows.index[rows.run != run].to_numpy()
            test = rows.index[rows.run == run].to_numpy()
            if set(d.loc[train, "run"]) & set(d.loc[test, "run"]):
                raise ValueError("Source run leaked across outer fold")
            yield str(run), train, test


def inner_folds(d, training, embargo=8.):
    rows = d.loc[training]
    runs = sorted(rows.run.unique())
    if len(runs) >= 2:
        for run in runs:
            yield rows.index[rows.run != run].to_numpy(), rows.index[rows.run == run].to_numpy()
        return
    ordered = rows.sort_values(["episode", "time", "row_id"])
    # With only one training block, fit earlier history and validate later.
    # Splits sharing an episode have an eight-second exclusion before validation.
    for fraction in (.5, .75):
        cut = int(len(ordered)*fraction)
        end = min(len(ordered), cut + max(1, len(ordered)//4))
        validation = ordered.iloc[cut:end]
        prior = ordered.iloc[:cut]
        if validation.empty or prior.empty:
            continue
        first = validation.iloc[0]
        prior = prior[(prior.episode != first.episode) | (prior.time < first.time-embargo)]
        if len(prior) and len(validation):
            yield prior.index.to_numpy(), validation.index.to_numpy()


def run_weights(d, indices):
    labels = d.loc[indices, "run"]
    counts = labels.value_counts()
    weights = labels.map(1./counts).to_numpy(float)
    return weights / weights.sum()


class Choice:
    """Full fixed class space, incumbent masked for destination-only likelihood.

    A small intercept penalty keeps absent-class logits finite. Absent training
    alternatives are reported, not silently removed or renumbered.
    """
    def fit(self, x, y, weights, classes, penalty, max_iter, tolerance, incumbent=None):
        if len(np.unique(y)) < 2:
            raise ValueError("training outcome has only one observed class")
        n, p = x.shape
        xa = np.column_stack((np.ones(n), x))
        regularizer = np.full((p+1, classes), 1./(penalty*n))
        regularizer[0] = 1e-6
        if incumbent is not None and np.any(y == incumbent):
            raise ValueError("Stay action appeared in conditional destination outcome")
        def evaluate(flat):
            beta = flat.reshape(p+1, classes)
            scores = xa @ beta
            if incumbent is not None:
                scores[np.arange(n), incumbent] = -1e15
            logs = scores-logsumexp(scores, axis=1, keepdims=True)
            objective = -weights @ logs[np.arange(n), y] + .5*np.sum(regularizer*beta**2)
            residual = np.exp(logs)
            residual[np.arange(n), y] -= 1.
            gradient = xa.T @ (weights[:, None]*residual)+regularizer*beta
            return objective, gradient.ravel()
        result = minimize(evaluate, np.zeros((p+1)*classes), jac=True, method="L-BFGS-B",
                          options=dict(maxiter=max_iter, ftol=tolerance*.01, gtol=tolerance, maxls=40))
        if not result.success:
            raise ValueError(f"nonconverged: {result.message}; iterations={result.nit}")
        self.beta = result.x.reshape(p+1, classes)
        self.iterations = int(result.nit)
        self.absent_classes = sorted(set(range(classes))-set(y))
        return self

    def predict(self, x, incumbent=None):
        scores = np.column_stack((np.ones(len(x)), x)) @ self.beta
        if incumbent is not None:
            scores[np.arange(len(x)), incumbent] = -1e15
        return np.exp(scores-logsumexp(scores, axis=1, keepdims=True))


class Design:
    def __init__(self, d, pixels, config, root):
        self.d, self.pixels, self.config, self.root = d, pixels, config, root
        self.root.mkdir(parents=True, exist_ok=True)

    def pixel_projection(self, train):
        ids = self.d.loc[train, "decision_id"].to_numpy(int)
        ids = np.unique(ids)
        key = C.digest(dict(train=ids, config=self.config, rows=len(self.pixels)))[:24]
        path = self.root / ("pca_"+key+".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                return z["projection"]
        selected = ids[np.linspace(0, len(ids)-1, min(len(ids), self.config["pca_fit_rows"])).astype(int)]
        sample = self.pixels[selected].astype(float)
        finite = np.isfinite(sample)
        means = np.divide(np.where(finite, sample, 0).sum(0), finite.sum(0),
                          out=np.zeros(sample.shape[1]), where=finite.sum(0)>0)
        sample = np.where(finite, sample, means)
        components = min(self.config["pca_components"], len(sample)-1, sample.shape[1])
        if components < 1:
            raise ValueError("Insufficient training rows for visual PCA")
        pca = PCA(n_components=components, svd_solver="randomized", random_state=self.config["seed"])
        pca.fit(sample)
        projection = np.empty((len(self.pixels), components), dtype=np.float32)
        for start in range(0, len(projection), 2048):
            x = self.pixels[start:start+2048]
            projection[start:start+2048] = pca.transform(np.where(np.isfinite(x), x, means))
        np.savez_compressed(path, projection=projection, training_decision_ids=ids,
                            components=pca.components_, imputation_mean=means, pca_mean=pca.mean_)
        return projection

    def prepare(self, train, test, spec):
        arrays = []; names = []
        for col in spec.columns:
            x = self.d[col].to_numpy(float)
            if col.startswith("previous_action_"):
                for a in range(-1, int(self.d.attrs["num_actions"])):
                    arrays.append((x == a).astype(float)); names.append(f"{col}={a}")
            else:
                if col.startswith(("root_p_", "actor_p_")):
                    x = np.log(np.clip(x, 1e-8, 1.))
                if col in ("bout_age", "budget", "event_index", "depth_pre"):
                    x = np.log1p(np.maximum(x, 0.))
                arrays.append(x); names.append(col)
        x = np.column_stack(arrays) if arrays else np.empty((len(self.d), 0))
        finite = np.isfinite(x)
        means = np.divide(np.where(finite[train], x[train], 0).sum(0), finite[train].sum(0),
                          out=np.zeros(x.shape[1]), where=finite[train].sum(0)>0)
        x = np.column_stack((np.where(finite, x, means), (~finite).astype(float)))
        names += [name+"_missing" for name in names.copy()]
        for source in ("bout_age", "budget", "event_index", "depth_pre"):
            if source not in spec.columns:
                continue
            col = spec.columns.index(source)
            # Locate the expanded numeric source (categorical one-hot widths differ).
            col = names.index(source)
            if len(np.unique(x[train, col])) >= 4:
                spline = SplineTransformer(n_knots=4, degree=3, include_bias=False, extrapolation="linear")
                spline.fit(x[train, col:col+1])
                block = spline.transform(x[:, col:col+1])
                x = np.column_stack((x, block))
                names += [source+f"_spline_{i}" for i in range(block.shape[1])]
        if spec.pixels:
            projected = self.pixel_projection(train)
            x = np.column_stack((x, projected[self.d.decision_id.to_numpy(int)]))
            names += [f"visual_pc_{i}" for i in range(projected.shape[1])]
        mean, sd = x[train].mean(0), x[train].std(0)
        keep = sd > 1e-10
        x = (x[:, keep]-mean[keep])/sd[keep]
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite preprocessed design")
        qa = dict(columns=[n for n, valid in zip(names, keep) if valid],
                  removed_constant=[n for n, valid in zip(names, keep) if not valid])
        return x[train], x[test], qa


class Engine:
    def __init__(self, args, decisions, pixels, cache_manifest, events=None):
        self.args, self.config = args, C.settings(args)
        self.n_actions = cache_manifest["num_actions"]
        self.signature = C.digest(dict(cache=cache_manifest["signature"], sources=C.sources(), config=self.config))
        self.root = args.output_root / "model_store" / C.tag(args) / self.signature[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.d = decisions.copy() if events is None else events.copy()
        self.d = self.d.reset_index(drop=True)
        self.d["row_id"] = np.arange(len(self.d))
        self.d.attrs["num_actions"] = self.n_actions
        self.design = Design(self.d, pixels, self.config, self.root / "transforms")
        self.is_event = events is not None

    def outcome(self, name):
        d = self.d
        eligible = d.scored.astype(bool) & (d.incumbent >= 0)
        if name == "switch":
            y, classes, conditional = d.switch.fillna(0).to_numpy(int), 2, False
        elif name == "destination":
            eligible &= d.switch == 1
            y, classes, conditional = d.human_action.to_numpy(int), self.n_actions, True
        elif name == "begin":
            y, classes, conditional = d.begin.to_numpy(int), 2, False
        elif name == "stop":
            # j=0 is initiation, not post-initiation continuation.
            eligible &= d.event_index > 0
            y, classes, conditional = d.stop_choice.to_numpy(int), 2, False
        else:
            raise ValueError(name)
        return y, classes, conditional, eligible.to_numpy()

    def fit(self, spec, outcome):
        check_control_features(spec, outcome)
        identity = C.digest(dict(spec=spec.__dict__, outcome=outcome, event=self.is_event))[:16]
        root = self.root / f"{outcome}_{spec.name}_{identity}"
        root.mkdir(exist_ok=True)
        ready = root / "completed.json"
        if ready.exists():
            return pd.read_csv(root / "predictions.csv.gz"), json.loads(ready.read_text())
        y, classes, conditional, eligible = self.outcome(outcome)
        incumbent = self.d.incumbent.to_numpy(int)
        records, failures, tuning, outer_qa = [], [], [], []
        for label, outer_train, outer_test in outer_folds(self.d):
            train, test = outer_train[eligible[outer_train]], outer_test[eligible[outer_test]]
            fold_path = root / (label+".csv.gz")
            qa_path = root / (label+".json")
            if fold_path.exists() and qa_path.exists():
                records.append(pd.read_csv(fold_path)); outer_qa.append(json.loads(qa_path.read_text())); continue
            print(f"fit {outcome}/{spec.name}: {label} train={len(train)} test={len(test)}", flush=True)
            if not len(test):
                failures.append(dict(fold=label, reason="no eligible test outcomes", train=len(train), test=0)); continue
            scores = {c: [] for c in self.config["c_grid"]}
            try:
                for inner_index, (it, iv) in enumerate(inner_folds(self.d, outer_train, self.config["embargo_seconds"])):
                    it, iv = it[eligible[it]], iv[eligible[iv]]
                    if not len(it) or not len(iv):
                        raise ValueError("empty eligible inner fold")
                    xt, xv, _ = self.design.prepare(it, iv, spec)
                    for penalty in scores:
                        try:
                            fit = Choice().fit(xt, y[it], run_weights(self.d, it), classes, penalty,
                                self.config["max_iter"], self.config["tolerance"], incumbent[it] if conditional else None)
                            probability = fit.predict(xv, incumbent[iv] if conditional else None)
                            lp = np.log(np.clip(probability[np.arange(len(iv)), y[iv]], 1e-15, 1.))
                            scores[penalty].append(float(run_weights(self.d, iv) @ lp))
                        except ValueError as error:
                            scores[penalty].append(np.nan)
                            tuning.append(dict(fold=label, inner=inner_index, C=penalty, reason=str(error)))
                usable = {c: np.mean(values) for c, values in scores.items() if values and np.isfinite(values).all()}
                if not usable:
                    raise ValueError("no regularization candidate valid in every inner fold")
                best = max(sorted(usable), key=lambda c: usable[c])
                xt, xv, design_qa = self.design.prepare(train, test, spec)
                model = Choice().fit(xt, y[train], run_weights(self.d, train), classes, best,
                    self.config["max_iter"], self.config["tolerance"], incumbent[train] if conditional else None)
                p = model.predict(xv, incumbent[test] if conditional else None)
                keep = [c for c in ("row_id", "decision_id", "event", "run", "session", "episode", "time", "incumbent", "human_action", "switch", "budget", "bout_age", "root_margin", "evidence_signed", "incumbent_change", "alternative_change", "event_index", "depth_pre") if c in self.d]
                pred = self.d.loc[test, keep].copy()
                pred["target"] = y[test]
                pred["logp"] = np.log(np.clip(p[np.arange(len(test)), y[test]], 1e-15, 1.))
                pred["brier"] = ((p[:, 1]-y[test])**2 if classes==2 and not conditional else
                                 np.sum((p-np.eye(classes)[y[test]])**2, axis=1))
                for a in range(classes): pred[f"p_{a}"] = p[:, a]
                pred.to_csv(fold_path, index=False)
                risk_file = None
                if conditional:
                    risk_test = outer_test[(self.d.loc[outer_test, "scored"].to_numpy(bool)) & (incumbent[outer_test]>=0)]
                    _, xrisk, _ = self.design.prepare(train, risk_test, spec)
                    risk_p = model.predict(xrisk, incumbent[risk_test])
                    risk = self.d.loc[risk_test, ["row_id", "decision_id", "run", "incumbent", "human_action"]].copy()
                    for a in range(classes): risk[f"p_{a}"] = risk_p[:, a]
                    risk_file = str(root/(label+".risk.csv.gz"))
                    risk.to_csv(risk_file, index=False)
                qa = dict(fold=label, session=int(self.d.loc[test[0], "session"]),
                          checkpoint=self.d.loc[test[0], "checkpoint"],
                          training_runs=sorted(self.d.loc[train, "run"].unique()),
                          train_rows=len(train), test_rows=len(test), C=best,
                          inner_scores=usable, iterations=model.iterations,
                          absent_training_classes=model.absent_classes, destination_risk_file=risk_file, **design_qa)
                C.write_json(qa_path, qa)
                outer_qa.append(qa); records.append(pred)
            except (ValueError, np.linalg.LinAlgError) as error:
                failures.append(dict(fold=label, reason=str(error), train=len(train), test=len(test)))
        columns = ["row_id", "decision_id", "run", "target", "logp", "brier"]
        prediction = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=columns)
        prediction.to_csv(root / "predictions.csv.gz", index=False)
        report = dict(spec=spec.__dict__, outcome=outcome, rows=len(prediction),
                      expected_rows=int(eligible.sum()), folds=outer_qa, failures=failures,
                      tuning_failures=tuning, cv="within-heldout-session-leave-one-source-block-out",
                      complete=not failures and len(prediction)==int(eligible.sum()))
        C.write_json(ready, report)
        return prediction, report


def compare(base, extended, base_qa, extended_qa, name, family, primary=""):
    joined = base[["row_id", "run", "logp", "brier"]].merge(
        extended[["row_id", "logp", "brier"]], on="row_id", suffixes=("_base", "_extended"), validate="one_to_one")
    joined["delta_ll"] = joined.logp_extended-joined.logp_base
    joined["delta_brier"] = joined.brier_base-joined.brier_extended
    by_run = joined.groupby("run", as_index=False).agg(delta_ll=("delta_ll", "mean"),
        delta_brier=("delta_brier", "mean"), rows=("row_id", "size"))
    reasons = []
    if not base_qa["complete"] or not extended_qa["complete"]:
        reasons.append("one or more outer folds not estimable; partial scores are descriptive only")
    added = set(extended_qa["spec"]["columns"])-set(base_qa["spec"]["columns"])
    if added and extended_qa["folds"] and all(
        not any(any(name == a or name.startswith(a+"_") for a in added) for name in f["columns"])
        for f in extended_qa["folds"]):
        reasons.append("all added predictors constant in all outer training folds")
    valid = bool(len(by_run) and not reasons)
    summary = dict(contrast=name, family=family, primary=primary, outcome=extended_qa["outcome"],
                   delta_ll=float(by_run.delta_ll.mean()) if valid else np.nan,
                   descriptive_delta_ll=float(by_run.delta_ll.mean()) if len(by_run) else np.nan,
                   delta_brier=float(by_run.delta_brier.mean()) if valid else np.nan,
                   runs=len(by_run), rows=len(joined), estimable=valid, reason="; ".join(reasons))
    return summary, by_run, joined
