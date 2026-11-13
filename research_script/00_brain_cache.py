#!/usr/bin/env python3
"""Prepare dynamic HRF regressors and verify independent BOLD reuse. outputs/00/neural/."""
import argparse
from common.paths import INPUT_CACHE,OUTPUTS
from common.features import prepare_unit,load_unit
from analyses.neural import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',type=int,required=True);p.add_argument('--game',type=int,required=True)
    p.add_argument('--cache-root',type=__import__('pathlib').Path,default=INPUT_CACHE)
    p.add_argument('--outputs-root',type=__import__('pathlib').Path,default=OUTPUTS)
    a=p.parse_args();prepare_unit(a.cache_root,a.outputs_root,a.subject,a.game)
    d,_=load_unit(a.outputs_root,a.subject,a.game,events=False)
    _,status=prepare(d,a.cache_root,a.outputs_root,a.subject,a.game);print(status)
