"""Session fixed effects and temporal covariance; no cross-run lag products.
Extracted from the validated session-FE implementation; unestimable effects stay NA.
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from . import native as base
LAGS = (64, 128, 256)
PRIMARY_LAG = 128
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
    audit["max_abs_coefficient"] = float(np.abs(coefficients).max())
    audit["information_condition_number"] = float(np.linalg.cond(hessian))
    if binary and audit["max_abs_coefficient"] > 50:
        return failure("unstable_large_coefficient")
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

