"""Preserved preprocessing entry for 00_prepare_harvard_oxford_masks.py."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.compat import export
export(globals(),"00_prepare_harvard_oxford_masks.py","00")
