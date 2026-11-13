"""Game-specific panel entry point, also available as numbered-script --panels."""
import argparse
from pathlib import Path
from common.paths import INPUT_CACHE,OUTPUTS


def execute(number,args):
    from .runner import run
    from .render import aggregate
    if args.aggregate:
        print(aggregate(args.outputs_root,list(range(1,7)),[args.game] if args.game is not None else [0,1,2],[number]).to_string(index=False))
    else:
        if args.subject is None or args.game is None:raise ValueError('--subject and --game are required')
        if number=='10':
            print(aggregate(args.outputs_root,list(range(1,7)),[args.game],['10']).to_string(index=False))
        elif number=='01':
            from .raw_behavior import run as raw_run
            print(raw_run(args.cache_root,args.outputs_root,args.subject,args.game))
        else:print(run(args.cache_root,args.outputs_root,args.subject,args.game))


def main(number):
    p=argparse.ArgumentParser(description=f'COSYNE panels for research theme {number}; game-specific outputs')
    p.add_argument('--subject',type=int,choices=range(1,7));p.add_argument('--game',type=int,choices=range(3))
    p.add_argument('--cache-root',type=Path,default=INPUT_CACHE);p.add_argument('--outputs-root',type=Path,default=OUTPUTS)
    p.add_argument('--aggregate',action='store_true');a=p.parse_args();execute(number,a)
