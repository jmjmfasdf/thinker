"""Same arguments and numbered output roots for every research theme."""
import argparse
from .paths import INPUT_CACHE,OUTPUTS

def main(number):
    from analyses.runner import run_unit
    from .reporting import aggregate
    parser=argparse.ArgumentParser(description=f'Research theme {number}; results go directly below outputs/{number}/')
    parser.add_argument('--subject',type=int,choices=range(1,7))
    parser.add_argument('--game',type=int,choices=range(3))
    parser.add_argument('--cache-root','--shared-cache-root',type=__import__('pathlib').Path,default=INPUT_CACHE)
    parser.add_argument('--outputs-root',type=__import__('pathlib').Path,default=OUTPUTS)
    parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--panels',action='store_true',help='Recompute/export the game-specific COSYNE panels; no neural analyses')
    parser.add_argument('--heatmaps',action='store_true',help='Theme02: smooth individual and equal-person M–U heatmaps for all six subjects')
    args=parser.parse_args()
    if args.heatmaps:
        if number!='02' or args.subject is not None:
            parser.error('--heatmaps is for theme02 and the complete six-subject cohort')
        from panels.mu_heatmaps import build
        result=build(args.outputs_root, games=[args.game] if args.game is not None else [0,1,2])
        # Keep existing theme02 artifact manifests valid for this standalone
        # command as well as for the full panel renderer.
        import json
        from common.io import sha,write_json
        from common.paths import output
        for game in result['games']:
            root=output(2,args.outputs_root)/'panels'/f'game{game}'
            path=root/'manifest.json'
            if path.exists():
                meta=json.loads(path.read_text())
                hm=json.loads((root/'heatmap/manifest.json').read_text())
                for name,digest in hm['figures'].items():
                    meta.setdefault('figures',{})[str(__import__('pathlib').Path(name).relative_to(root))]=digest
                meta['heatmap_manifest_sha256']=sha(root/'heatmap/manifest.json')
                write_json(path,meta)
        print(result['status']);return
    if args.panels:
        if number in ['06','07','08']:
            parser.error('Neural panels are excluded from this campaign')
        from panels.cli import execute
        execute(number,args);return
    if number=='10':
        from analyses.optional import utility
        print(utility(args.outputs_root));return
    if args.aggregate:
        aggregate(number,args.outputs_root,list(range(1,7)),[args.game] if args.game is not None else list(range(3)));return
    if args.subject is None or args.game is None:parser.error('--subject and --game are required')
    print(run_unit(number,args.cache_root,args.outputs_root,args.subject,args.game))
