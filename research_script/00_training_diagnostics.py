#!/usr/bin/env python3
"""Audit recorded 10M checkpoint bindings; does not launch training or score evaluation."""
import argparse
from pathlib import Path
import json
import pandas as pd
import yaml
from common.paths import INPUT_CACHE,OUTPUTS,output
from common.io import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',type=Path,default=INPUT_CACHE);p.add_argument('--outputs-root',type=Path,default=OUTPUTS)
    args=p.parse_args();rows=[]
    for binding in args.cache_root.glob('checkpoint_leave*/sub*/checkpoint_binding.json'):
        b=json.loads(binding.read_text());config=Path(b['checkpoint'])/'config_c.yaml'
        flags=yaml.safe_load(config.read_text()) if config.exists() else {}
        rows.append(dict(subject=b['subject'],game=b['game'],leave_session=b['leave_session'],
            checkpoint_real_step=b['checkpoint_real_step'],checkpoint=b['checkpoint'],
            temperature=flags.get('voc_gate_q_temperature'),gru_connected=flags.get('voc_gru_connected'),
            exact_projection=flags.get('voc_gate_exact_projection'),status=b['status']))
    root=output(0,args.outputs_root)/'training';root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(root/'checkpoint_summary.csv',index=False)
    write_json(root/'manifest.json',dict(status='COMPLETED',checkpoints=len(rows),source=str(args.cache_root),
        meaning='Bound checkpoint step, not an inference from folder name; training curves remain in logs/thinker'))
