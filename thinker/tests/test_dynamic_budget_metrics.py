import torch

import thinker.buffer as buffer_module
from thinker import util
from tests.test_dynamic_cenv import _flags


class _CaptureWriter:
    def __init__(self, **_kwargs):
        self.rows = []

    def log(self, stats):
        self.rows.append(dict(stats))


def _insert(buffer, search_steps, stage_end):
    time_n, batch_n = search_steps.shape
    reward_n = len(util.get_reward_names(buffer.flags))
    return buffer.insert(
        step_status=torch.zeros(time_n, batch_n, dtype=torch.long),
        episode_return=torch.zeros(time_n, batch_n, reward_n),
        episode_step=torch.zeros(time_n, batch_n, dtype=torch.long),
        real_done=torch.zeros(time_n, batch_n, dtype=torch.bool),
        actor_id=torch.arange(batch_n).unsqueeze(0),
        real_transition=torch.zeros(time_n, batch_n, dtype=torch.bool),
        stage_end=stage_end,
        forced_stop=torch.zeros(time_n, batch_n, dtype=torch.bool),
        search_steps=search_steps,
    )


def test_self_play_budget_metrics_cover_empty_and_completed_batches(
        tmp_path, monkeypatch):
    monkeypatch.setattr(buffer_module, "FileWriter", _CaptureWriter)
    flags = _flags(cap=4)
    flags.self_play_n = 1
    flags.env_n = 3
    flags.ckp = False
    flags.savedir = str(tmp_path)
    flags.xpid = "budget-metrics"
    flags.ckpdir = str(tmp_path / flags.xpid)
    raw_class = buffer_module.SelfPlayBuffer.__ray_metadata__.modified_class
    self_play_buffer = raw_class(flags)

    # Row zero is the overlapping bootstrap row and must never contribute.
    _insert(
        self_play_buffer,
        search_steps=torch.tensor([[999, 999, 999], [7, 8, 9]]),
        stage_end=torch.zeros(2, 3, dtype=torch.bool),
    )
    first_stats = self_play_buffer.plogger.rows[-1]
    assert first_stats["max_budget"] == 0.0
    assert first_stats["mean_budget"] == 0.0

    _insert(
        self_play_buffer,
        search_steps=torch.tensor([
            [999, 999, 999],
            [1, 4, 99],
            [0, 88, 2],
        ]),
        stage_end=torch.tensor([
            [True, True, True],
            [True, True, False],
            [True, False, True],
        ]),
    )
    stats = self_play_buffer.plogger.rows[-1]
    assert stats["max_budget"] == 4.0
    assert stats["mean_budget"] == 1.75

