"""Read-only 0--10M gradient and coverage summary for the candidate review."""
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_script/outputs/00/planning_candidate_analysis_20261114'
RUN=ROOT/'logs/thinker/enduro-sub001-all-min1-c0001-tau001-gru-episode-10m-s701-20261111'


def main():
    path=RUN/'logs.csv'
    with path.open('rb') as f:blob=f.read(path.stat().st_size)
    blob=blob[:blob.rfind(b'\n')+1]
    wanted=['real_step','search/wait_fraction','search/real_transition_fraction',
        'search/forced_stop_count','collector/forced_stop_count',
        'gradient/voc_gru_candidate_norm','gradient/voc_gru_applied_norm',
        'gradient/actor_gru_gradient_norm','gradient/voc_actor_gru_cosine',
        'gradient/gru_update_norm','gradient/actor_optimizer_stepped']
    d=pd.read_csv(io.BytesIO(blob),comment='#',usecols=lambda x:x in wanted)
    d['norm_ratio']=d['gradient/voc_gru_applied_norm']/d['gradient/actor_gru_gradient_norm'].replace(0,np.nan)
    d['interval']=pd.cut(d.real_step,[-1,2e6,4e6,6e6,8e6,np.inf],labels=['0–2M','2–4M','4–6M','6–8M','8–10M'])
    d.to_csv(OUT/'historical_gradient_rows.csv',index=False)
    summary=d.groupby('interval',observed=True).agg(
        updates=('real_step','size'),start=('real_step','min'),end=('real_step','max'),
        median_voc_to_actor_norm=('norm_ratio','median'),p10_voc_to_actor_norm=('norm_ratio',lambda x:x.quantile(.1)),
        p90_voc_to_actor_norm=('norm_ratio',lambda x:x.quantile(.9)),
        median_cosine=('gradient/voc_actor_gru_cosine','median'),
        opposed_fraction=('gradient/voc_actor_gru_cosine',lambda x:(x<0).mean()),
        median_voc_norm=('gradient/voc_gru_applied_norm','median'),
        median_other_actor_norm=('gradient/actor_gru_gradient_norm','median'),
        median_wait_fraction=('search/wait_fraction','median'),
        median_real_transition_fraction=('search/real_transition_fraction','median'))
    summary.to_csv(OUT/'historical_gradient_summary.csv')
    d['bin']=np.minimum((d.real_step//250000).astype(int),39)
    b=d.groupby('bin').agg(step=('real_step','median'),median=('norm_ratio','median'),
        p10=('norm_ratio',lambda x:x.quantile(.1)),p90=('norm_ratio',lambda x:x.quantile(.9)),
        cosine=('gradient/voc_actor_gru_cosine','median'))
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(b.step/1e6,b['median']*100)
    ax[0].fill_between(b.step/1e6,b.p10*100,b.p90*100,alpha=.2)
    ax[0].set(yscale='log',ylabel='VOC / other actor GRU gradient norm (%)',xlabel='Training real environment steps (M)',title='Actual training batches: median and 10–90%')
    ax[1].plot(b.step/1e6,b.cosine);ax[1].axhline(0,color='gray',linestyle='--')
    ax[1].set(xlabel='Training real environment steps (M)',ylabel='Median gradient cosine',title='VOC alignment with the other actor gradient')
    fig.tight_layout();fig.savefig(OUT/'historical_gradients.png',dpi=180);fig.savefig(OUT/'historical_gradients.pdf');plt.close(fig)
    meta=dict(source=str(path),source_sha256=hashlib.sha256(blob).hexdigest(),last_step=int(d.real_step.max()),
        updates=len(d),optimizer_skipped=int((~d['gradient/actor_optimizer_stepped'].astype(bool)).sum()),
        nonfinite_ratio=int((~np.isfinite(d.norm_ratio)).sum()),
        collector_forced_stop_max=float(d['collector/forced_stop_count'].max()),
        learner_forced_stop_total=float(d['search/forced_stop_count'].sum()),
        metric='Norm ratio before clipping; distinct from the marginal Adam update measured by offline probes.',
        limitation='Temporally correlated training updates; cannot infer VOC target correctness from magnitude or direction.')
    (OUT/'historical_gradient_manifest.json').write_text(json.dumps(meta,indent=2))
    print(summary.to_string());print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
