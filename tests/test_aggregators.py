from __future__ import annotations

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

from fed_agent.fed.aggregators import (
    coordinate_median_state_dict,
    fedavg_state_dict,
    state_dict_nbytes,
    trimmed_mean_state_dict,
)


def test_fedavg_state_dict_and_bytes() -> None:
    a = OrderedDict({"w": torch.ones(2, dtype=torch.float32)})
    b = OrderedDict({"w": torch.zeros(2, dtype=torch.float32)})
    out = fedavg_state_dict([a, b], weights=[1.0, 3.0])
    assert torch.allclose(out["w"], torch.tensor([0.25, 0.25], dtype=torch.float32))
    assert state_dict_nbytes(out) > 0


def test_coordinate_median_ignores_outlier() -> None:
    # Three "good" clients near 1.0 and one poisoned client at 100 -> median
    # must stay near 1.0, unlike the mean which the outlier would drag up.
    good = [OrderedDict({"w": torch.ones(3, dtype=torch.float32)}) for _ in range(3)]
    bad = OrderedDict({"w": torch.full((3,), 100.0, dtype=torch.float32)})
    out = coordinate_median_state_dict(good + [bad])
    assert torch.allclose(out["w"], torch.ones(3, dtype=torch.float32))


def test_trimmed_mean_drops_extremes() -> None:
    sds = [
        OrderedDict({"w": torch.tensor([0.0], dtype=torch.float32)}),
        OrderedDict({"w": torch.tensor([1.0], dtype=torch.float32)}),
        OrderedDict({"w": torch.tensor([1.0], dtype=torch.float32)}),
        OrderedDict({"w": torch.tensor([100.0], dtype=torch.float32)}),
    ]
    # trim 1 per side (0.25 * 4 = 1): drops 0.0 and 100.0 -> mean(1.0, 1.0) = 1.0
    out = trimmed_mean_state_dict(sds, trim_ratio=0.25)
    assert torch.allclose(out["w"], torch.tensor([1.0], dtype=torch.float32))
    # trim_ratio=0 reduces to plain mean.
    out0 = trimmed_mean_state_dict(sds, trim_ratio=0.0)
    assert torch.allclose(out0["w"], torch.tensor([25.5], dtype=torch.float32))
