"""Descriptive first-root-branch concentration; no new hypothesis tests."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import OUTPUTS,INPUT_CACHE,tag
from common.io import write_json,sha
DEST=OUTPUTS/'02/value_update_candidates_20261111'


def unit(arg):
    subject,game=arg
    d=pd.read_csv(DEST/tag(subject,game)/'candidate_decisions.csv.gz')
    d=d.loc[d.scored.eq(1)&d.incumbent_action.ge(0)]
    rows=[]
    for cp in range(1,5):
        r=d.loc[d.session.eq(cp)]
        path=INPUT_CACHE/f'checkpoint_leave{cp:02d}'/tag(subject,game)/'persistence_events.csv.gz'
        e=pd.read_csv(path,usecols=['real_step_id','status2_position','branch'])
        e=e.loc[e.real_step_id.isin(r.real_step_id)&e.branch.ge(0)]
        first=e.sort_values('status2_position').drop_duplicates('real_step_id').merge(r[['real_step_id','run_label']],on='real_step_id',validate='one_to_one')
        for run,z in first.groupby('run_label'):
            counts=z.branch.value_counts();freq=counts/counts.sum()
            rows.append(dict(subject=subject,game=game,checkpoint=cp,run_label=run,n_decisions=len(z),
                dominant_first_branch=int(counts.index[0]),dominant_fraction=float(freq.iloc[0]),
                first_branch_entropy=float(-(freq*np.log(freq)).sum()),n_distinct_first_branches=len(counts)))
    return pd.DataFrame(rows)


def main():
    with ProcessPoolExecutor(max_workers=3) as pool:
        r=pd.concat(pool.map(unit,[(s,g) for s in range(1,7) for g in range(3)]),ignore_index=True)
    r.to_csv(DEST/'first_branch_concentration_runs.csv',index=False)
    s=r.groupby(['subject','game'],as_index=False)[['dominant_fraction','first_branch_entropy']].mean()
    s.to_csv(DEST/'first_branch_concentration_subjects.csv',index=False)
    g=s.groupby('game').agg(mean=('dominant_fraction','mean'),min_subject=('dominant_fraction','min'),max_subject=('dominant_fraction','max'))
    g.to_csv(DEST/'first_branch_concentration_group.csv')
    write_json(DEST/'first_branch_concentration_manifest.json',dict(status='COMPLETED',source_sha256=sha(__file__),
        definition='Most frequent first root branch within each run, among held-out scored decisions with a sampled branch. Equal runs then subjects. Descriptive only.'))
    print(g.to_string(),flush=True)


if __name__=='__main__':main()
