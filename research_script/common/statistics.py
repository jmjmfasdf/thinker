"""Group inference uses participants, never decisions or checkpoint replicates."""
import itertools
import numpy as np
import pandas as pd
from scipy import stats

def group_inference(rows, keys, expected_subjects=6):
    rows = rows.copy()
    # Equal-run averaging happens before this function for prediction/matching.
    units = rows.groupby([*keys, "subject"], dropna=False).value.mean().reset_index()
    records, vectors = [], []
    subjects = sorted(units.subject.unique())
    for key, part in units.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        s = part.set_index("subject").value.reindex(subjects)
        v = s.to_numpy(float); good = np.isfinite(v); n = int(good.sum())
        r = dict(zip(keys, key), n_subjects=n, mean=float(np.nanmean(v)) if n else np.nan,
                 sem=float(stats.sem(v[good])) if n > 1 else np.nan, p=np.nan, p_max_t=np.nan,
                 status="ok" if n == expected_subjects else "incomplete_subjects")
        if n == expected_subjects:
            signs = np.array(list(itertools.product([-1, 1], repeat=n)))
            r["p"] = float(np.mean(np.abs(signs @ v / n) >= abs(v.mean())-1e-12))
        records.append(r); vectors.append(v)
    if not records: return pd.DataFrame()
    # A fixed complete family is required; never silently reduce correction to successful fits.
    if len(subjects) == expected_subjects and all(r["status"] == "ok" for r in records):
        matrix = np.column_stack(vectors)
        signs = np.array(list(itertools.product([-1,1], repeat=len(subjects))))
        perm = signs[:,:,None] * matrix[None,:,:]
        t = np.abs(perm.mean(1)) / np.maximum(perm.std(1,ddof=1)/np.sqrt(len(subjects)),1e-12)
        obs = np.abs(matrix.mean(0))/np.maximum(matrix.std(0,ddof=1)/np.sqrt(len(subjects)),1e-12)
        for i,r in enumerate(records): r["p_max_t"] = float(np.mean(t.max(1) >= obs[i]-1e-12))
    return pd.DataFrame(records)
