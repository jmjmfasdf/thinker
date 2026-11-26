"""Explicit reproduction of old leave-session results in a separate destination.

The preserved implementation has its original scientific numbering. It is
never selected by current numbered scripts and must not write current outputs.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import os
from .paths import REPO,OUTPUTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['legacy_leave_session'],required=True)
    p.add_argument('--legacy-theme',choices=['02','03','04','05','06','07','08','09','12'],required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--subject',type=int,required=True);p.add_argument('--game',type=int,choices=[0,1,2],required=True)
    p.add_argument('--outputs-root',type=Path,required=True)
    a=p.parse_args()
    if a.outputs_root.resolve()==OUTPUTS.resolve() or a.outputs_root.resolve().is_relative_to(OUTPUTS.resolve()):
        p.error('Legacy comparison requires a separate output root outside current outputs')
    source=REPO/'legacy/research_script/pre_reorganization/source/research_script'
    if not source.is_dir():raise FileNotFoundError(source)
    code='from analyses.runner import run_unit; import sys; print(run_unit(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5])))'
    process=subprocess.run([sys.executable,'-B','-c',code,a.legacy_theme,str(a.cache_root.resolve()),
        str(a.outputs_root.resolve()),str(a.subject),str(a.game)],cwd=source,
        env=dict(os.environ,THINKER_REPOSITORY_ROOT=str(REPO)),check=False)
    raise SystemExit(process.returncode)
