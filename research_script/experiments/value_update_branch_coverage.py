"""Descriptive support audit for the value-update screen; no new tests."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import sys
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import OUTPUTS,INPUT_CACHE,tag
from common.io import sha,write_json

DEST=OUTPUTS/'02/value_update_candidates_20261111'


def unit(arg):
    subject,game=arg
    d=pd.read_csv(DEST/tag(subject,game)/'candidate_decisions.csv.gz')
    d=d.loc[d.scored.eq(1)&d.incumbent_action.ge(0)].copy()
    idcols=['real_step_id','incumbent_action','initial_alternative']
    parts=[];sources={}
    for cp in range(1,5):
        r=d.loc[d.session.eq(cp)]
        p=INPUT_CACHE/f'checkpoint_leave{cp:02d}'/tag(subject,game)/'persistence_events.csv.gz'
        e=pd.read_csv(p,usecols=['real_step_id','branch','value_revision_valid','root_q_revision'])
        e=e.merge(r[idcols],on='real_step_id',how='inner',validate='many_to_one')
        valid=e.value_revision_valid.eq(1)&np.isfinite(e.root_q_revision)
        # Counts are observations of a branch, separately from valid repeated updating.
        e['ongoing_query']=e.branch.eq(e.incumbent_action)
        e['ongoing_revised']=e.ongoing_query&valid
        e['initial_alt_query']=e.branch.eq(e.initial_alternative)
        e['initial_alt_revised']=e.initial_alt_query&valid
        e['any_revision']=valid
        flags=e.groupby('real_step_id')[['ongoing_query','ongoing_revised','initial_alt_query','initial_alt_revised','any_revision']].any()
        branches=e.loc[e.branch.ge(0)].groupby('real_step_id').branch.nunique()
        flags=flags.join(branches.rename('distinct_root_branches'))
        parts.append(flags);sources[str(p)]=sha(p)
    flags=pd.concat(parts)
    if flags.index.has_duplicates:raise ValueError('Duplicate held-out decision')
    d=d.merge(flags,on='real_step_id',how='left',validate='one_to_one')
    cols=['ongoing_query','ongoing_revised','initial_alt_query','initial_alt_revised','any_revision']
    d[cols]=d[cols].fillna(False).astype(bool)
    d['distinct_root_branches']=d.distinct_root_branches.fillna(0).astype(int)
    assert d.any_revision.eq(d.valid_update_count.gt(0)).all()
    rows=[]
    for label,r in [('all',d),('previous_noop',d.loc[d.incumbent_action.eq(0)]),('previous_active',d.loc[d.incumbent_action.ne(0)])]:
        for run,z in r.groupby('run_label'):
            search=z.additional_computations.gt(0)
            row=dict(subject=subject,game=game,population=label,run_label=run,n=len(z),n_searched=int(search.sum()),
                zero_budget_fraction=float((~search).mean()),zero_U_fraction=float(z.U_all.eq(0).mean()),
                multiple_root_branches_fraction=float(z.distinct_root_branches.ge(2).mean()),
                multiple_branches_given_search=float(z.loc[search,'distinct_root_branches'].ge(2).mean()))
            for name in cols:
                row[name+'_fraction']=float(z[name].mean())
                row[name+'_given_search']=float(z.loc[search,name].mean())
            rows.append(row)
    return pd.DataFrame(rows),sources


def main():
    results=[];sources={}
    with ProcessPoolExecutor(max_workers=3) as pool:
        for r,src in pool.map(unit,[(s,g) for s in range(1,7) for g in range(3)]):
            results.append(r);sources.update(src)
    runs=pd.concat(results,ignore_index=True)
    cols=[c for c in runs if c.endswith(('fraction','given_search'))]
    subjects=runs.groupby(['subject','game','population'],as_index=False)[cols].mean()
    group=subjects.groupby(['game','population'],as_index=False)[cols].mean()
    for name,r in [('branch_coverage_runs',runs),('branch_coverage_subjects',subjects),('branch_coverage_group',group)]:
        r.to_csv(DEST/(name+'.csv'),index=False)
    write_json(DEST/'branch_coverage_manifest.json',dict(status='COMPLETED',sources=sources,source_sha256=sha(__file__),
        interpretation='Descriptive support audit only; no additional hypothesis tests. Equal-run then equal-subject averages. Query and repeated revision are distinct.'))
    print(group.to_string(index=False),flush=True)


if __name__=='__main__':main()
