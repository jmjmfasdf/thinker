#!/usr/bin/env python3
"""Prepare latest native model regressors against independently verified BOLD.

Only alignment and HRF preprocessing; no neural association is fitted here.
"""
from pathlib import Path
import json
import pandas as pd
from common.research_cli import parser
from common.research_inputs import prepare_native, sha, BlockedInput
from common.result_store import ResultStore
from common.io import clean
from analyses.neural_current import prepare_current, input_fingerprint


def prepare_brain(args, subject, game):
    if args.render_only or args.stage=='render':
        raise ValueError('00 brain preparation has no figures; --render-only / --stage render are unsupported')
    prepared=prepare_native(args,subject,game);root=Path(prepared['cache'])
    saved=json.loads((root/'logs/run.json').read_text())
    with ResultStore(root,saved['identity'],resume=True) as store:
        qa=None
        try:
            with store.stage('analyze',dict(code=sha(__file__),native=prepared['metadata']['manifest_sha256'],
                    reader=sha(Path(__file__).parent/'analyses/neural_current.py'),
                    neural_inputs=input_fingerprint(subject,game))) as stage:
                if not stage.reused:
                    runs,qa=prepare_current(prepared['decisions'],args.input_root,subject,game)
                    if qa.get('status')=='BLOCKED_INPUT':raise BlockedInput(qa.get('reason',str(qa)))
                    columns=[]
                    for i,run in enumerate(runs):
                        name=f'neural_design_{i:03d}'
                        stage.write_arrays(name,x=run['x'])
                        columns.extend(dict(array=name,run=run['run'],index=j,column=c) for j,c in enumerate(run['names']))
                    stage.write_table('neural_design_columns',pd.DataFrame(columns,columns=['array','run','index','column']))
            if qa is not None:
                store.meta['neural_preparation']=clean(qa)
                store.log('Neural preparation QA: '+json.dumps(clean(qa),ensure_ascii=False,sort_keys=True))
            store.publish(required_stages=('prepare','analyze'))
        except FileNotFoundError as error:
            store.blocked_input(str(error))
            return dict(subject=subject,game=game,status='BLOCKED_INPUT',reason=str(error),path=str(root))
    return dict(subject=subject,game=game,status=store.meta['status'],
                scientific_status=str(store.meta.get('neural_preparation',{}).get('status','UNSPECIFIED')).upper(),path=str(root))


def main():
    p=parser('00');args=p.parse_args()
    if args.render_only or args.stage=='render':p.error('00 brain preparation has no figures; render-only is unsupported')
    if 'mario' in args.games:p.error('Mario has no current model-derived neural preparation')
    results=[]
    for subject in args.subjects:
        for game in args.games:
            try:result=prepare_brain(args,subject,game)
            except Exception as error:
                result=dict(subject=subject,game=game,status='BLOCKED_INPUT' if isinstance(error,FileNotFoundError) else 'FAILED',
                            error_type=type(error).__name__,reason=str(error))
            results.append(result);print(json.dumps(result),flush=True)
    if any(r['status']=='FAILED' for r in results):raise SystemExit(1)
    if any(r['status']=='BLOCKED_INPUT' for r in results):raise SystemExit(2)


if __name__=='__main__':main()
