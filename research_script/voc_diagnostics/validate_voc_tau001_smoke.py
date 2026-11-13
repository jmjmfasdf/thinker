"""Validate three-game full-geometry smoke and prepare RTX 2080 Ti utility probes."""
import csv
import json
import math
from pathlib import Path
import sys

from voc10m_parallel_monitor import atomic,publish


def main():
    campaign=Path(sys.argv[1]).resolve()
    results={}
    for game in ('pong','enduro','spaceinvaders'):
        cell={'name':'smoke-t001-'+game,'game':game}
        run=campaign/'smoke/cells'/cell['name']
        rows=list(csv.DictReader((run/'logs.csv').open()))
        assert rows and all(None not in row for row in rows)
        assert int(float(rows[-1]['real_step']))>=6000
        assert float(rows[-1]['training/bc_targets_committed'])>0
        for key in ('gradient/voc_gru_applied_norm','gradient/gru_update_norm'):
            values=[float(r[key]) for r in rows]
            assert all(math.isfinite(v) for v in values),key
            assert any(v>0 for v in values),key
        metrics=json.loads((run/'collector_metrics.json').read_text())
        assert metrics['terminal'] is True and metrics['physical_env_steps']>=6000
        assert (run/'finish').exists()
        assert (run/'control_samples.jsonl').stat().st_size>0
        bundle=publish(campaign/'smoke',cell,4000)
        assert bundle is not None
        import yaml
        config=yaml.safe_load((run/'config_c.yaml').read_text())
        assert config['voc_gate_q_temperature']==.01
        assert config['voc_gate_exact_projection'] and config['voc_gru_connected']
        assert config['env_n']==16 and config['model_batch_size']==32
        results[game]={'bundle':str(bundle),'real_step':int(float(rows[-1]['real_step'])),
                       'bc_targets':float(rows[-1]['training/bc_targets_committed']),
                       'temperature':config['voc_gate_q_temperature'],'collector':metrics}
        atomic(campaign/f'smoke_benefit_{game}.json',dict(cell=cell['name'],game=game,
            real_step=4000,kind='benefit',smoke=True,bundle=str(bundle),
            output=str(campaign/'smoke_evaluations'/game)))
    atomic(campaign/'smoke_validation.json',{'status':'PASS','results':results})
    print('THREE_GAME_TAU001_TRAINING_AND_CHECKPOINT_VALIDATION_PASS',flush=True)


if __name__=='__main__':main()
