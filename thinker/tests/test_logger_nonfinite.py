from types import SimpleNamespace

import math

from thinker.logger import SLogWorker


def test_parse_line_preserves_nonfinite_diagnostics_instead_of_dropping_them():
    worker = SLogWorker.__new__(SLogWorker)
    worker.real_step = 12
    worker._logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)

    row = worker.parse_line(
        ["tick", "loss", "positive", "negative", "finite"],
        "7,nan,inf,-inf,0.25",
    )

    assert row["tick"] == 7
    assert row["_tick"] == 7
    assert math.isnan(row["loss"])
    assert row["positive"] == float("inf")
    assert row["negative"] == float("-inf")
    assert row["finite"] == 0.25
