#!/usr/bin/env python3
"""Recorded final REAL-head likelihood and accuracy on held-out human sessions."""
import argparse
from pathlib import Path
import pandas as pd
from common.paths import INPUT_CACHE,OUTPUTS,output,tag
from common.features import prepare_unit,load_unit
from common.prediction import scores
from common.reporting import save_tables
from common.io import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',type=int,required=True);p.add_argument('--game',type=int,required=True)
    p.add_argument('--cache-root',type=Path,default=INPUT_CACHE);p.add_argument('--outputs-root',type=Path,default=OUTPUTS)
    a=p.parse_args();prepare_unit(a.cache_root,a.outputs_root,a.subject,a.game)
    d,_=load_unit(a.outputs_root,a.subject,a.game,events=False)
    d=d.loc[d.scored.eq(1)&d.actor_action_probability_valid.eq(1)]
    n=int(d.n_actions.iloc[0]);rows=[]
    for run,r in d.groupby('run_label'):
        probs=r[[f'actor_action_probability_{j}' for j in range(n)]].to_numpy()
        rows.append(dict(run_label=run,subject=a.subject,game=a.game,**scores(r.human_action,probs)))
    root=output(1,a.outputs_root)/'model_fidelity'/tag(a.subject,a.game)
    save_tables(root,{'run_scores':pd.DataFrame(rows)})
    write_json(root/'manifest.json',dict(status='COMPLETED',heldout=True,policy='Recorded REAL-head probabilities; no second softmax'))

if __name__=='__main__':main()
