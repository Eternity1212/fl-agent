from __future__ import annotations

import torch
import torch.nn as nn

from fed_agent.models.retfound_lora import LoRALinear, inject_lora


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 6)
        self.head = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.head(self.qkv(x))


def test_inject_lora_replaces_target_linear() -> None:
    model = _TinyBlock()
    n = inject_lora(model, target_keywords=("qkv",), rank=2, alpha=4.0)
    assert n == 1
    assert isinstance(model.qkv, LoRALinear)
    assert model(torch.zeros(1, 4)).shape == (1, 2)
    assert any("lora_" in name for name, _p in model.named_parameters())
