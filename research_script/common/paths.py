"""Canonical numbered outputs and portable repository paths."""
from pathlib import Path
import os

REPO = Path(os.environ.get('THINKER_REPOSITORY_ROOT', Path(__file__).resolve().parents[2]))
SCRIPTS = REPO / 'research_script'
OUTPUTS = SCRIPTS / "outputs"
INPUT_CACHE = OUTPUTS / "00_dynamic_cache_10m_20261110"
GAMES = {0: "Enduro", 1: "Pong", 2: "Space Invaders"}
THEMES = {
    "00": "cache_and_training_diagnostics", "01": "behavior_and_model_fidelity",
    "02": "action_boundary_mechanisms", "03": "dynamic_computation_control",
    "04": "search_to_action", "05": "human_prediction", "06": "neural_decoding",
    "07": "neural_rsa", "08": "neural_encoding", "09": "task_state_complexity",
    "10": "computation_utility", "12": "continuous_dynamics", "13": "behavioral_modes",
}

def output(number, root=OUTPUTS):
    number = f"{int(number):02d}"
    if number not in THEMES:
        raise ValueError(f"Unknown theme: {number}")
    return Path(root) / number

def tag(subject, game):
    return f"sub{subject:03d}_game{game}"
