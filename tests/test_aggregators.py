from __future__ import annotations

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

from fed_agent.fed.aggregators import fedavg_state_dict, state_dict_nbytes


def test_fedavg_state_dict_and_bytes() -> None:
    a = OrderedDict({"w": torch.ones(2, dtype=torch.float32)})
    b = OrderedDict({"w": torch.zeros(2, dtype=torch.float32)})
    out = fedavg_state_dict([a, b], weights=[1.0, 3.0])
    assert torch.allclose(out["w"], torch.tensor([0.25, 0.25], dtype=torch.float32))
    assert state_dict_nbytes(out) > 0
