"""Within-stage probability changes, preserving complete recorded decision units."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_script/outputs/00/planning_candidate_analysis_20261114'


def main():
    q=pd.read_csv(OUT/'support_v2/optional_rows.csv')
    stages=pd.read_csv(OUT/'support_v2/complete_stages.csv')
    keys=['attempt','stream','decision']
    q=q.merge(stages[keys],on=keys,validate='many_to_one')
    rows=[]
    for key,g in q.groupby(keys):
        g=g.sort_values('count');p=g.p_continue.to_numpy();chosen=g['continue'].to_numpy(bool);p0=p[0]
        sequential=np.log(np.where(chosen,p,1-p)).sum()
        frozen=np.log(np.where(chosen,p0,1-p0)).sum()
        rows.append(dict(attempt=key[0],stream=key[1],decision=key[2],real_step=g.real_step.iloc[-1],
            choices=len(g),first_p=p0,last_p=p[-1],probability_range=np.ptp(p),
            max_shift_from_first=np.max(np.abs(p-p0)),sequential_logscore=sequential,
            frozen_logscore=frozen,logscore_gain=sequential-frozen))
    d=pd.DataFrame(rows);d['interval']=pd.cut(d.real_step,[-1,2e6,4e6,6e6,8e6,np.inf],labels=['0–2M','2–4M','4–6M','6–8M','8–10M'])
    d.to_csv(OUT/'gate_within_stage_rows.csv',index=False)
    summary=d.groupby('interval',observed=True).agg(decisions=('decision','size'),
        mean_logscore_gain=('logscore_gain','mean'),median_max_probability_shift=('max_shift_from_first','median'),
        probability_shift_over_005_fraction=('max_shift_from_first',lambda x:(x>.05).mean()))
    late=d[(d.interval=='8–10M')&(d.choices>=8)]
    summary.to_csv(OUT/'gate_within_stage_summary.csv')
    (OUT/'gate_within_stage_manifest.json').write_text(json.dumps(dict(
        long_latest_decisions=len(late),mean_probability_range=float(late.probability_range.mean()),
        mean_logscore_gain=float(late.logscore_gain.mean()),
        caution='Descriptive execution likelihood, not human action likelihood or environmental utility; observed complete stages only.'),indent=2))
    print(summary.to_string());print('long stages latest',late[['probability_range','logscore_gain']].mean().to_dict())


if __name__=='__main__':main()
