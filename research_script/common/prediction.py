"""Fixed regularization, training-only preprocessing, session-held-out scoring."""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from .native import prepare_matrix

def probabilities(d, columns, target, train, test, *, soft=False, actions=None):
    if train.sum() < 100 or not test.any(): return None, {"status":"insufficient_training_or_test"}
    x, names = prepare_matrix(d, list(dict.fromkeys(columns)), train)
    audit = dict(n_train=int(train.sum()), n_test=int(test.sum()), n_features=len(names))
    if soft:
        target = np.asarray(target, float)
        if not np.isfinite(target[train]).all(): return None, dict(audit,status="invalid_soft_target")
        a = target.shape[1]
        xx = np.column_stack([np.ones(len(x)), x]); xt=xx[train]; y=target[train]
        def objective(flat):
            beta=flat.reshape(xx.shape[1],a-1)
            logits=np.column_stack([np.zeros(len(xt)),xt@beta])
            lp=logits-logsumexp(logits,axis=1,keepdims=True)
            loss=-(y*lp).sum()/len(y)+0.0005*(beta[1:]**2).sum()
            grad=xt.T@(np.exp(lp)[:,1:]-y[:,1:])/len(y)
            grad[1:]+=0.001*beta[1:]
            return loss,grad.ravel()
        fit=minimize(objective,np.zeros(xx.shape[1]*(a-1)),jac=True,method="L-BFGS-B",
                     options={"maxiter":1000,"ftol":1e-10,"gtol":1e-6})
        audit.update(iterations=int(fit.nit),gradient_max=float(np.abs(fit.jac).max()))
        if not fit.success or audit["gradient_max"] > 1e-4:
            return None,dict(audit,status="nonconverged",message=str(fit.message))
        logits=np.column_stack([np.zeros(test.sum()),xx[test]@fit.x.reshape(xx.shape[1],a-1)])
        return np.exp(logits-logsumexp(logits,axis=1,keepdims=True)),dict(audit,status="ok")
    y=np.asarray(target)
    if len(np.unique(y[train]))<2:return None,dict(audit,status="single_training_class")
    # The exact-action multinomial has highly correlated nested evidence blocks.
    # Newton's explicit Hessian converges more reliably than a 1,000-step L-BFGS cap.
    model=LogisticRegression(C=1.,solver="newton-cholesky",max_iter=200,tol=1e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always");model.fit(x[train],y[train])
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
        return None,dict(audit,status="nonconverged")
    p=model.predict_proba(x[test]); out=np.zeros((test.sum(),actions or 2))
    out[:,model.classes_.astype(int)]=p
    return out,dict(audit,status="ok",iterations=int(model.n_iter_.max()))

def scores(truth, p):
    truth=np.asarray(truth,dtype=int);selected=p[np.arange(len(p)),truth]
    result=dict(n=len(truth),log_score=float(np.log(np.maximum(selected,1e-12)).mean()),
                accuracy=float((p.argmax(1)==truth).mean()),zero_probability_count=int((selected==0).sum()))
    if p.shape[1]==2:
        result["brier"]=float(((p[:,1]-truth)**2).mean())
        result["auroc"]=float(roc_auc_score(truth,p[:,1])) if len(np.unique(truth))==2 else np.nan
    return result

def baseline(d):
    return ["log_age","log_age_sq","recent10","recent30","prior_reward","prior_done",
            "run_progress","run_progress_sq"]+[c for c in d if c.startswith(("previous_action_","visual_")) and c!='visual_change']

def blocks(d):
    b0=baseline(d)
    b1=b0+["M","root_value","root_entropy","initial_q_margin"]+[c for c in d if c.startswith("log_policy_")]
    b2=b1+["log_budget","max_depth","searched","carried"]
    b3=b2+["U_all","K","signed_update_sum","terminal_return_gain"]
    b3 += [c for c in d if c.startswith("root_qs_mean_terminal_")]
    return {"B0_behavior_visual_history":b0,"B1_initial_support":b1,"B2_quantity":b2,
            "B3_search_content":b3,"B4_VOC":b3+["ema_initial","ema_change"]}
