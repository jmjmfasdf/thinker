"""Sequential research themes, independently parallel subject/game units.

Uses the same CLI and result transactions as every numbered entrypoint.
No training, environment evaluation or visualization generation is launched.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.research_cli import THEMES


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--themes',nargs='+',default=[f'{n:02d}' for n in range(1,11)])
    own,rest=p.parse_known_args()
    if '--help' in rest:
        print('Add --themes 01 02 ... 13. Other options are shared with numbered scripts.\n')
        from common.research_cli import parser
        parser('01').print_help();return
    failures=[]
    for n in own.themes:
        number=f'{int(n):02d}'
        if number not in THEMES:raise ValueError('Unknown theme '+number)
        target=ROOT/f'{number}_{THEMES[number]}.py'
        process=subprocess.run([sys.executable,'-B','-u',str(target),*rest],check=False)
        if process.returncode:failures.append((number,process.returncode))
    if failures:
        print('Incomplete themes:',failures);raise SystemExit(1)

if __name__=='__main__':main()
