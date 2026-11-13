"""Forecast errors become predictors only after their observations arrive."""
from pathlib import Path
import numpy as np
import pandas as pd
from common.paths import tag

def attach_arrived_mismatch(d,cache,subject,game):
    d=d.copy();d['arrived_reward_error']=np.nan;d['arrived_error_available']=0.
    for cp,part in d.groupby('checkpoint'):
        path=Path(cache)/f'checkpoint_leave{int(cp):02d}'/tag(subject,game)/'forecast_outcomes.csv.gz'
        cols=['run_label','episode_id','outcome_available_time','outcome_available_event',
              'state_comparison_primary','reward_prediction_error']
        forecasts=pd.read_csv(path,usecols=cols)
        forecasts=forecasts.loc[forecasts.state_comparison_primary.eq(1)&
            np.isfinite(forecasts.reward_prediction_error)&np.isfinite(forecasts.outcome_available_time)&
            np.isfinite(forecasts.outcome_available_event)]
        groups={(run,ep):f for (run,ep),f in forecasts.groupby(['run_label','episode_id'])}
        for key,s in part.groupby(['run_label','episode_id']):
            if key not in groups:continue
            s=s.sort_values('decision_time');f=groups[key]
            times=s.decision_time.to_numpy();events=s.first_control_native_event.to_numpy()
            ix=np.searchsorted(times,f.outcome_available_time.to_numpy(),side='left')
            valid=ix<len(s);ix=ix[valid];f=f.loc[valid]
            # An outcome at the same wall time must precede this model control event.
            available=f.outcome_available_event.to_numpy()
            ix=np.maximum(ix,np.searchsorted(events,available,side='left'))
            valid=ix<len(s);ix=ix[valid];f=f.loc[valid]
            sums=np.zeros(len(s));counts=np.zeros(len(s))
            np.add.at(sums,ix,np.abs(f.reward_prediction_error.to_numpy()));np.add.at(counts,ix,1)
            arrival=pd.Series(np.divide(sums,counts,out=np.full(len(s),np.nan),where=counts>0)).ffill().to_numpy()
            d.loc[s.index,'arrived_reward_error']=arrival
            d.loc[s.index,'arrived_error_available']=np.isfinite(arrival).astype(float)
    return d
