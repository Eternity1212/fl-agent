"""Reusable centralized/local/federated runner for RFMiD paper experiments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fed_agent.data.rfmid import load_rfmid_label_table
from fed_agent.data.rfmid_torch import RFMiDTorchDataset
from fed_agent.fed.aggregators import fedavg_state_dict
from fed_agent.fed.simulator import TinyCNN, TinyMLP
from fed_agent.metrics.multilabel import multilabel_classification_metrics
from fed_agent.models.retfound_lora import ModelBuildInfo, build_retfound_lora_model
from fed_agent.splits.partition import read_split_json


@dataclass(frozen=True)
class PaperRunConfig:
    method: str
    backbone: str = "retfound_mae_vit_large"
    seed: int = 0
    epochs: int = 1
    rounds: int = 1
    local_epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-3
    fedprox_mu: float = 0.0
    loss: str = "bce"
    positive_dropout: float = 0.0
    train_label_noise: float = 0.0
    image_size: tuple[int, int] = (224, 224)
    device: str = "cuda"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    require_retfound: bool = False
    # --- throughput / multi-GPU options ---
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    use_data_parallel: bool = False
    # --- agent (adaptive orchestration) options ---
    agent_aggregation: str = "agent"  # "agent" | "size" (size = FedAvg baseline)
    agent_tau: float = 0.05
    agent_geometry: bool = False
    agent_noisy_clients: tuple[int, ...] = ()  # client ids that get label noise
    agent_client_noise: float = 0.0  # symmetric flip prob for noisy clients
    # joint adaptive orchestration: per-client proximal strength from telemetry
    agent_adaptive_mu: bool = False
    agent_mu_max: float = 0.1
    agent_mu_tau: float = 0.05


class _InMemoryDataset(torch.utils.data.Dataset):
    """Cache resized tensors once for fast smoke/fallback matrices."""

    def __init__(self, base: RFMiDTorchDataset) -> None:
        self._base = base
        self._items = [base[i] for i in range(len(base))]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return self._base.image_ids

    def indices_for_ids(self, ids: list[str]) -> list[int]:
        return self._base.indices_for_ids(ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        x, y, meta = self._items[index]
        return x.clone(), y.clone(), dict(meta)


def _pos_weight(labels_csv: Path) -> torch.Tensor | None:
    _ids, _names, y = load_rfmid_label_table(labels_csv)
    pos = torch.tensor(y.sum(axis=0), dtype=torch.float32)
    neg = torch.tensor(float(y.shape[0]), dtype=torch.float32) - pos
    w = torch.where(pos > 0, neg / torch.clamp(pos, min=1.0), torch.ones_like(pos))
    return torch.clamp(w, min=1.0, max=20.0)


def _make_model(
    cfg: PaperRunConfig,
    *,
    n_labels: int,
    in_dim: int,
) -> tuple[nn.Module, dict[str, Any]]:
    if cfg.backbone == "mlp":
        model = TinyMLP(in_dim=in_dim, n_labels=n_labels)
        info = {"actual_backbone": "mlp", "is_retfound": False}
        return model, info
    if cfg.backbone == "tiny_cnn":
        model = TinyCNN(n_labels=n_labels)
        info = {"actual_backbone": "tiny_cnn", "is_retfound": False}
        return model, info
    model, build_info = build_retfound_lora_model(
        n_labels=n_labels,
        backbone=cfg.backbone,
        lora_rank=int(cfg.lora_rank),
        lora_alpha=float(cfg.lora_alpha),
        require_retfound=bool(cfg.require_retfound),
    )
    return model, asdict(build_info) if isinstance(build_info, ModelBuildInfo) else dict(build_info)


def _apply_positive_dropout(y: torch.Tensor, *, p: float, seed: int) -> torch.Tensor:
    if p <= 0.0:
        return y
    g = torch.Generator(device=y.device)
    g.manual_seed(int(seed))
    mask = (y > 0.5) & (torch.rand(y.shape, generator=g, device=y.device) < float(p))
    out = y.clone()
    out[mask] = 0.0
    return out


def _apply_symmetric_flip(y: torch.Tensor, *, p: float, seed: int) -> torch.Tensor:
    """Flip binary labels 0<->1 with probability p (symmetric label noise)."""
    if p <= 0.0:
        return y
    g = torch.Generator(device=y.device)
    g.manual_seed(int(seed))
    flip = torch.rand(y.shape, generator=g, device=y.device) < float(p)
    out = y.clone()
    out[flip] = 1.0 - out[flip]
    return out


def _loss_fn(cfg: PaperRunConfig, pos_weight: torch.Tensor | None) -> nn.Module:
    if cfg.loss == "balanced_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if cfg.loss == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss: {cfg.loss}")


def _device_for(cfg: PaperRunConfig) -> torch.device:
    return torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")


def _loader_kwargs(cfg: PaperRunConfig) -> dict[str, Any]:
    num_workers = max(int(cfg.num_workers), 0)
    pin_memory = bool(cfg.pin_memory) and _device_for(cfg).type == "cuda"
    kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    return kwargs


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _maybe_wrap_data_parallel(model: nn.Module, cfg: PaperRunConfig) -> nn.Module:
    dev = _device_for(cfg)
    if dev.type != "cuda" or not bool(cfg.use_data_parallel):
        return model
    if torch.cuda.device_count() < 2:
        return model
    return nn.DataParallel(model)


def _state_dict_cpu(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """CPU snapshot of the (unwrapped) model params. ``.clone()`` is required for
    correctness on CPU, where ``.cpu()`` is a no-op that would otherwise alias
    live parameters; it is a harmless extra copy on CUDA."""
    base = _unwrap_model(model)
    return OrderedDict({k: v.detach().cpu().clone() for k, v in base.state_dict().items()})


def _load_state_dict(model: nn.Module, state_dict: OrderedDict[str, torch.Tensor]) -> None:
    _unwrap_model(model).load_state_dict(state_dict)


def _train_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    cfg: PaperRunConfig,
    pos_weight: torch.Tensor | None,
    global_vec: torch.Tensor | None = None,
    round_seed: int = 0,
    label_flip_p: float = 0.0,
    mu_override: float | None = None,
) -> float:
    device = _device_for(cfg)
    model.to(device)
    model.train()
    pw = pos_weight.to(device) if pos_weight is not None else None
    loss_fn = _loss_fn(cfg, pw)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg.lr))
    mu = float(mu_override) if mu_override is not None else float(cfg.fedprox_mu)
    losses: list[float] = []
    for _epoch in range(int(cfg.local_epochs)):
        for x, y, _meta in loader:
            x = x.to(device)
            y = y.to(device)
            if label_flip_p > 0.0:
                y = _apply_symmetric_flip(y, p=float(label_flip_p), seed=int(round_seed))
            y = _apply_positive_dropout(y, p=float(cfg.positive_dropout), seed=int(round_seed))
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            if mu > 0.0 and global_vec is not None:
                cur = torch.nn.utils.parameters_to_vector(list(model.parameters()))
                loss = loss + 0.5 * mu * (cur - global_vec).pow(2).sum()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(len(losses), 1))


def _predict(model: nn.Module, loader: DataLoader, *, device: str) -> tuple[np.ndarray, np.ndarray]:
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.to(dev)
    model.eval()
    ys: list[torch.Tensor] = []
    ps: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y, _meta in loader:
            logits = model(x.to(dev))
            ys.append(y.cpu())
            ps.append(torch.sigmoid(logits).cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def _trainable_nbytes(model: nn.Module) -> int:
    return int(sum(p.detach().cpu().numpy().nbytes for p in model.parameters() if p.requires_grad))


def _prepare(
    *,
    train_labels_csv: Path,
    train_images_dir: Path,
    eval_labels_csv: Path,
    eval_images_dir: Path,
    cfg: PaperRunConfig,
) -> tuple[RFMiDTorchDataset, DataLoader, nn.Module, dict[str, Any], torch.Tensor | None]:
    train_noise = float(cfg.train_label_noise)
    train_ds = RFMiDTorchDataset(
        labels_csv=train_labels_csv,
        images_dir=train_images_dir,
        image_size=cfg.image_size,
        label_noise_p_flip=train_noise if train_noise > 0.0 else None,
        label_noise_seed=int(cfg.seed),
    )
    eval_ds = RFMiDTorchDataset(
        labels_csv=eval_labels_csv,
        images_dir=eval_images_dir,
        image_size=cfg.image_size,
    )
    if max(cfg.image_size) <= 64:
        train_ds = _InMemoryDataset(train_ds)  # type: ignore[assignment]
        eval_ds = _InMemoryDataset(eval_ds)  # type: ignore[assignment]
    x0, y0, _m = train_ds[0]
    model, info = _make_model(cfg, n_labels=int(y0.shape[0]), in_dim=int(x0.numel()))
    model = _maybe_wrap_data_parallel(model, cfg)
    eval_loader = DataLoader(
        eval_ds, batch_size=int(cfg.batch_size), shuffle=False, **_loader_kwargs(cfg)
    )
    pw = _pos_weight(train_labels_csv) if cfg.loss == "balanced_bce" else None
    return train_ds, eval_loader, model, info, pw


def _build_eval_loader(
    labels_csv: Path,
    images_dir: Path,
    *,
    cfg: PaperRunConfig,
) -> DataLoader:
    """Build a non-shuffled loader (for probe / eval) matching _prepare semantics."""
    ds: Any = RFMiDTorchDataset(
        labels_csv=labels_csv,
        images_dir=images_dir,
        image_size=cfg.image_size,
    )
    if max(cfg.image_size) <= 64:
        ds = _InMemoryDataset(ds)
    return DataLoader(
        ds, batch_size=int(cfg.batch_size), shuffle=False, **_loader_kwargs(cfg)
    )


def _eval_payload(
    model: nn.Module,
    eval_loader: DataLoader,
    *,
    cfg: PaperRunConfig,
) -> dict[str, Any]:
    y, p = _predict(model, eval_loader, device=cfg.device)
    return multilabel_classification_metrics(y, p, calibrate=True)


def _run_centralized(
    *,
    train_ds: RFMiDTorchDataset,
    eval_loader: DataLoader,
    model: nn.Module,
    cfg: PaperRunConfig,
    pos_weight: torch.Tensor | None,
) -> dict[str, Any]:
    loader = DataLoader(
        train_ds, batch_size=int(cfg.batch_size), shuffle=True, **_loader_kwargs(cfg)
    )
    losses: list[float] = []
    for e in range(int(cfg.epochs)):
        losses.append(_train_model(model, loader, cfg=cfg, pos_weight=pos_weight, round_seed=e))
    return {"train_loss": losses, "eval": _eval_payload(model, eval_loader, cfg=cfg)}


def _run_federated(
    *,
    train_ds: RFMiDTorchDataset,
    eval_loader: DataLoader,
    model: nn.Module,
    split_json: Path,
    cfg: PaperRunConfig,
    pos_weight: torch.Tensor | None,
) -> dict[str, Any]:
    split = read_split_json(split_json)
    clients: dict[str, list[str]] = split["clients"]
    client_keys = sorted(clients.keys(), key=lambda x: int(x))
    global_sd = _state_dict_cpu(model)
    dev = _device_for(cfg)
    round_losses: list[float] = []
    upload_bytes: list[int] = []
    for r in range(int(cfg.rounds)):
        local_sds = []
        weights = []
        losses = []
        global_vec = (
            torch.nn.utils.parameters_to_vector(list(_unwrap_model(model).parameters()))
            .detach()
            .clone()
            .to(dev)
        )
        for ck in client_keys:
            idxs = train_ds.indices_for_ids(list(clients[ck]))
            if not idxs:
                continue
            _load_state_dict(model, global_sd)
            subset = Subset(train_ds, idxs)
            loader = DataLoader(
                subset, batch_size=int(cfg.batch_size), shuffle=True, **_loader_kwargs(cfg)
            )
            loss = _train_model(
                model,
                loader,
                cfg=cfg,
                pos_weight=pos_weight,
                global_vec=global_vec if cfg.method in {"fedprox", "robust_fedprox"} else None,
                round_seed=int(cfg.seed) + r * 1000 + int(ck),
            )
            local_sds.append(_state_dict_cpu(model))
            weights.append(float(len(idxs)))
            losses.append(loss)
        global_sd = fedavg_state_dict(local_sds, weights)
        _load_state_dict(model, global_sd)
        round_losses.append(float(sum(losses) / max(len(losses), 1)))
        upload_bytes.append(_trainable_nbytes(model) * len(local_sds))
    return {
        "train_loss": round_losses,
        "comm_bytes_upload_per_round": upload_bytes,
        "total_upload_bytes": int(sum(upload_bytes)),
        "eval": _eval_payload(model, eval_loader, cfg=cfg),
    }


def _run_agent_federated(
    *,
    train_ds: RFMiDTorchDataset,
    eval_loader: DataLoader,
    model: nn.Module,
    split_json: Path,
    cfg: PaperRunConfig,
    pos_weight: torch.Tensor | None,
    probe_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Federated training with adaptive (agent) or size-based aggregation.

    Supports per-client label noise (``agent_noisy_clients``) so that the agent's
    selective down-weighting can be evaluated against an equal/size-weighted
    FedAvg baseline under identical conditions.
    """
    from fed_agent.agent.orchestrator import decide_client_mu, decide_weights

    split = read_split_json(split_json)
    clients: dict[str, list[str]] = split["clients"]
    client_keys = sorted(clients.keys(), key=lambda x: int(x))
    noisy = {int(c) for c in cfg.agent_noisy_clients}
    global_sd = _state_dict_cpu(model)
    dev = _device_for(cfg)
    use_prox = (cfg.fedprox_mu > 0.0) or bool(cfg.agent_adaptive_mu)
    need_scores = (cfg.agent_aggregation == "agent") or bool(cfg.agent_adaptive_mu)
    round_losses: list[float] = []
    upload_bytes: list[int] = []
    weight_history: list[dict[str, float]] = []
    probe_history: list[dict[str, float]] = []
    mu_history: list[dict[str, float]] = []
    prev_scores: dict[str, float] = {}
    for r in range(int(cfg.rounds)):
        local_sds = []
        sizes: list[float] = []
        cks: list[str] = []
        losses = []
        global_vec = (
            torch.nn.utils.parameters_to_vector(list(_unwrap_model(model).parameters()))
            .detach()
            .clone()
            .to(dev)
        )
        # Per-client proximal strength from the PREVIOUS round's probe scores.
        cur_mu: dict[str, float] = {}
        if cfg.agent_adaptive_mu and prev_scores:
            ordered = [k for k in client_keys if k in prev_scores]
            mus = decide_client_mu(
                probe_scores=[prev_scores[k] for k in ordered],
                mu_max=float(cfg.agent_mu_max),
                tau=float(cfg.agent_mu_tau),
            )
            cur_mu = {k: m for k, m in zip(ordered, mus)}
        for ck in client_keys:
            idxs = train_ds.indices_for_ids(list(clients[ck]))
            if not idxs:
                continue
            _load_state_dict(model, global_sd)
            subset = Subset(train_ds, idxs)
            loader = DataLoader(
                subset, batch_size=int(cfg.batch_size), shuffle=True, **_loader_kwargs(cfg)
            )
            flip = float(cfg.agent_client_noise) if int(ck) in noisy else 0.0
            mu_i = cur_mu.get(ck, 0.0)
            loss = _train_model(
                model,
                loader,
                cfg=cfg,
                pos_weight=pos_weight,
                global_vec=global_vec if use_prox else None,
                round_seed=int(cfg.seed) + r * 1000 + int(ck),
                label_flip_p=flip,
                mu_override=mu_i if cfg.agent_adaptive_mu else None,
            )
            local_sds.append(_state_dict_cpu(model))
            sizes.append(float(len(idxs)))
            cks.append(ck)
            losses.append(loss)
        mu_history.append({ck: float(cur_mu.get(ck, 0.0)) for ck in cks})

        probe_scores: list[float] = []
        if need_scores:
            scorer = probe_loader if probe_loader is not None else eval_loader
            for sd in local_sds:
                _load_state_dict(model, sd)
                y, p = _predict(model, scorer, device=cfg.device)
                # Negative validation BCE: more sensitive to label-noise damage
                # than AUROC, so clean clients score clearly higher than noisy.
                eps = 1e-7
                pc = np.clip(p, eps, 1.0 - eps)
                bce = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))
                probe_scores.append(-bce)
            probe_history.append({ck: float(s) for ck, s in zip(cks, probe_scores)})
            prev_scores = {ck: float(s) for ck, s in zip(cks, probe_scores)}

        if cfg.agent_aggregation == "agent":
            decision = decide_weights(
                probe_scores=probe_scores,
                sizes=sizes,
                tau=float(cfg.agent_tau),
            )
            weights = decision.weights
        else:
            weights = list(sizes)

        weight_history.append({ck: float(w) for ck, w in zip(cks, weights)})
        global_sd = fedavg_state_dict(local_sds, weights)
        _load_state_dict(model, global_sd)
        round_losses.append(float(sum(losses) / max(len(losses), 1)))
        upload_bytes.append(_trainable_nbytes(model) * len(local_sds))

    return {
        "train_loss": round_losses,
        "comm_bytes_upload_per_round": upload_bytes,
        "total_upload_bytes": int(sum(upload_bytes)),
        "agent_weight_history": weight_history,
        "agent_probe_history": probe_history,
        "agent_mu_history": mu_history,
        "agent_noisy_clients": sorted(noisy),
        "agent_probe_heldout": bool(probe_loader is not None),
        "agent_adaptive_mu": bool(cfg.agent_adaptive_mu),
        "eval": _eval_payload(model, eval_loader, cfg=cfg),
    }


def _run_local_only(
    *,
    train_ds: RFMiDTorchDataset,
    eval_loader: DataLoader,
    model_info: dict[str, Any],
    split_json: Path,
    cfg: PaperRunConfig,
    pos_weight: torch.Tensor | None,
    n_labels: int,
    in_dim: int,
) -> dict[str, Any]:
    split = read_split_json(split_json)
    metrics: list[dict[str, Any]] = []
    for ck, ids in sorted(split["clients"].items(), key=lambda x: int(x[0])):
        idxs = train_ds.indices_for_ids(list(ids))
        if not idxs:
            continue
        model, _info = _make_model(cfg, n_labels=n_labels, in_dim=in_dim)
        model = _maybe_wrap_data_parallel(model, cfg)
        subset = Subset(train_ds, idxs)
        loader = DataLoader(
            subset, batch_size=int(cfg.batch_size), shuffle=True, **_loader_kwargs(cfg)
        )
        for e in range(int(cfg.epochs)):
            _train_model(model, loader, cfg=cfg, pos_weight=pos_weight, round_seed=e + int(ck))
        metrics.append(_eval_payload(model, eval_loader, cfg=cfg))
    keys = ["best_macro_f1_present", "best_micro_f1", "macro_f1_present", "micro_f1"]
    mean_eval = {k: float(sum(float(m[k]) for m in metrics) / max(len(metrics), 1)) for k in keys}
    return {"eval": mean_eval, "n_clients": len(metrics), "model_info": model_info}


def run_paper_experiment(
    *,
    train_labels_csv: Path,
    train_images_dir: Path,
    eval_labels_csv: Path,
    eval_images_dir: Path,
    cfg: PaperRunConfig,
    split_json: Path | None = None,
    probe_labels_csv: Path | None = None,
    probe_images_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one centralized/local/federated paper experiment."""

    torch.manual_seed(int(cfg.seed))
    train_ds, eval_loader, model, info, pos_weight = _prepare(
        train_labels_csv=train_labels_csv,
        train_images_dir=train_images_dir,
        eval_labels_csv=eval_labels_csv,
        eval_images_dir=eval_images_dir,
        cfg=cfg,
    )
    probe_loader = None
    if probe_labels_csv is not None and probe_images_dir is not None:
        probe_loader = _build_eval_loader(probe_labels_csv, probe_images_dir, cfg=cfg)
    x0, y0, _m = train_ds[0]
    if cfg.method == "centralized":
        result = _run_centralized(
            train_ds=train_ds,
            eval_loader=eval_loader,
            model=model,
            cfg=cfg,
            pos_weight=pos_weight,
        )
    elif cfg.method == "local_only":
        if split_json is None:
            raise ValueError("local_only requires split_json")
        result = _run_local_only(
            train_ds=train_ds,
            eval_loader=eval_loader,
            model_info=info,
            split_json=split_json,
            cfg=cfg,
            pos_weight=pos_weight,
            n_labels=int(y0.shape[0]),
            in_dim=int(x0.numel()),
        )
    elif cfg.method in {"fedavg", "fedprox", "robust_fedprox"}:
        if split_json is None:
            raise ValueError(f"{cfg.method} requires split_json")
        result = _run_federated(
            train_ds=train_ds,
            eval_loader=eval_loader,
            model=model,
            split_json=split_json,
            cfg=cfg,
            pos_weight=pos_weight,
        )
    elif cfg.method == "agent_fed":
        if split_json is None:
            raise ValueError("agent_fed requires split_json")
        result = _run_agent_federated(
            train_ds=train_ds,
            eval_loader=eval_loader,
            probe_loader=probe_loader,
            model=model,
            split_json=split_json,
            cfg=cfg,
            pos_weight=pos_weight,
        )
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    return {"config": asdict(cfg), "model_info": info, **result}
