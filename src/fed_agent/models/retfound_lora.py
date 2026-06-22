"""RETFound / ViT model factory with lightweight LoRA adapters.

The official RETFound checkpoints are distributed through Hugging Face and may be
gated. This module separates **access detection** from model construction so the
experiment runner can fail loudly instead of silently reporting fallback results
as RETFound results.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

RETFOUND_HF_REPO = os.environ.get("RETFOUND_HF_REPO", "YukunZhou/RETFound_mae_natureCFP")
RETFOUND_CKPT_FILENAME = os.environ.get("RETFOUND_CKPT_FILENAME", "RETFound_mae_natureCFP.pth")


def resolve_hf_token() -> str | None:
    """Read a Hugging Face access token from common environment variables."""

    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    return None


@dataclass(frozen=True)
class ModelBuildInfo:
    """Metadata persisted with every paper experiment run."""

    requested_backbone: str
    actual_backbone: str
    is_retfound: bool
    checkpoint_path: str | None
    lora_rank: int
    lora_alpha: float
    trainable_params: int
    total_params: int


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper for ``nn.Linear``.

    The base layer is frozen; only ``lora_a`` and ``lora_b`` are trainable.
    """

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, int(rank), bias=False)
        self.lora_b = nn.Linear(int(rank), base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def _get_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    target_keywords: Iterable[str] = ("qkv", "attn.proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> int:
    """Replace matching Linear modules with LoRA wrappers and return count."""

    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(k in name for k in target_keywords):
            targets.append(name)

    for name in targets:
        parent, child = _get_parent(model, name)
        base = getattr(parent, child)
        setattr(
            parent,
            child,
            LoRALinear(base, rank=int(rank), alpha=float(alpha), dropout=float(dropout)),
        )
    return len(targets)


def _set_classifier(model: nn.Module, n_labels: int) -> None:
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=int(n_labels))
        return
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = int(model.head.in_features)
        model.head = nn.Linear(in_features, int(n_labels))
        return
    raise ValueError("Could not find classifier head on model")


def _load_retfound_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    model.load_state_dict(filtered, strict=False)


def check_retfound_access(
    *,
    repo_id: str = RETFOUND_HF_REPO,
    filename: str = RETFOUND_CKPT_FILENAME,
) -> tuple[bool, str]:
    """Return ``(ok, message_or_path)`` for gated RETFound checkpoint access."""

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=filename, token=resolve_hf_token())
        return True, str(path)
    except Exception as exc:  # pragma: no cover - depends on user token/network
        return False, f"{type(exc).__name__}: {exc}"


def _param_counts(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def build_retfound_lora_model(
    *,
    n_labels: int,
    backbone: str = "retfound_mae_vit_large",
    fallback_backbone: str = "vit_base_patch16_224",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    require_retfound: bool = False,
) -> tuple[nn.Module, ModelBuildInfo]:
    """Build RETFound+LoRA if checkpoint is accessible, otherwise explicit timm fallback."""

    try:
        import timm
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Install paper extras: python3 -m pip install -e '.[paper]'") from exc

    is_retfound = False
    ckpt_path: str | None = None
    actual = fallback_backbone
    model_name = fallback_backbone

    if backbone.startswith("retfound"):
        ok, msg = check_retfound_access()
        if ok:
            ckpt_path = msg
            is_retfound = True
            actual = "retfound_mae_vit_large"
            model_name = "vit_large_patch16_224"
        elif require_retfound:
            raise RuntimeError(f"RETFound checkpoint is not accessible: {msg}")

    model = timm.create_model(model_name, pretrained=not is_retfound, num_classes=int(n_labels))
    _set_classifier(model, int(n_labels))
    if ckpt_path is not None:
        _load_retfound_checkpoint(model, Path(ckpt_path))

    for p in model.parameters():
        p.requires_grad_(False)
    _set_classifier(model, int(n_labels))
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if hasattr(classifier, "parameters"):
            for p in classifier.parameters():
                p.requires_grad_(True)
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad_(True)

    n_lora = inject_lora(
        model,
        rank=int(lora_rank),
        alpha=float(lora_alpha),
        dropout=float(lora_dropout),
    )
    if n_lora == 0:
        raise RuntimeError("No LoRA target modules were found; inspect model.named_modules()")

    trainable, total = _param_counts(model)
    info = ModelBuildInfo(
        requested_backbone=backbone,
        actual_backbone=actual,
        is_retfound=is_retfound,
        checkpoint_path=ckpt_path,
        lora_rank=int(lora_rank),
        lora_alpha=float(lora_alpha),
        trainable_params=trainable,
        total_params=total,
    )
    return model, info
