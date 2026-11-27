"""Canonical numbered outputs and portable repository paths."""
from pathlib import Path
import os

REPO = Path(os.environ.get('THINKER_REPOSITORY_ROOT', Path(__file__).resolve().parents[2]))
SCRIPTS = REPO / 'research_script'
OUTPUTS = SCRIPTS / "outputs"
INPUT_CACHE = OUTPUTS / "00_dynamic_cache_10m_20261110"
GAMES = {0: "Enduro", 1: "Pong", 2: "Space Invaders"}
THEMES = {'00': 'native_preparation', '01': 'behavioral_analysis', '02': 'model_validation', '03': 'computation_allocation', '04': 'computation_utility', '05': 'evaluation_revision', '06': 'search_to_policy', '07': 'computation_to_behavior', '08': 'evaluation_continuity', '09': 'evaluation_accumulation', '10': 'control_patterns', '11': 'neural_encoding', '12': 'neural_decoding', '13': 'neural_rsa'}


def output(number, root=OUTPUTS):
    number = f"{int(number):02d}"
    if number not in THEMES:
        raise ValueError(f"Unknown theme: {number}")
    return Path(root) / number

def output_base(args):
    """Both sample modes replace the same public numbered result.

    Publication policy is separate from sampling.py so path-only changes do not
    invalidate the independently verified 00 extraction of both sample views.
    """
    return Path(args.outputs_root)


def tag(subject, game):
    return f"sub{subject:03d}_game{game}"
