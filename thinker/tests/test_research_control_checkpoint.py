"""10 imports the owning checkpoint runtime before native rich extraction."""
from argparse import Namespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'research_script'))
from analyses import control_pattern_pipeline as pipeline


def test_control_pattern_loads_frozen_buffer_before_extraction(tmp_path, monkeypatch):
    owner = tmp_path / 'capture_campaign'
    checkpoint = owner / 'checkpoints/run/milestones/10000000'
    calls = []

    def audit(args, game, out):
        calls.append('audit')
        pipeline.native.write_json(out / 'audit.json', {'checkpoint': str(checkpoint)})

    def frozen(root, module):
        assert root == owner and module == 'buffer'
        calls.append('buffer')

    class ReachedExtraction(Exception):
        pass

    def extract(args, game, out):
        assert calls == ['audit', 'buffer']
        raise ReachedExtraction

    monkeypatch.setattr(pipeline.native, 'audit_game', audit)
    monkeypatch.setattr(pipeline.native, 'frozen_import', frozen)
    monkeypatch.setattr(pipeline.native, 'extract_game', extract)
    with pytest.raises(ReachedExtraction):
        pipeline.analyze(Namespace(), 1, 'pong', {}, tmp_path / 'work')
