from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fed_agent.data.rfmid_torch import RFMiDTorchDataset
from fed_agent.fed.aggregators import fedavg_state_dict, state_dict_nbytes
from fed_agent.noise.protocol import load_noise_protocol_yaml, parse_noise_protocol_v1
from fed_agent.splits.partition import read_split_json


@dataclass(frozen=True)
class FedSmokeConfig:
    rounds: int = 2
    local_epochs: int = 1
    batch_size: int = 2
    lr: float = 0.05
    fedprox_mu: float = 0.0
    device: str = "cpu"


class TinyMLP(nn.Module):
    """A tiny MLP baseline for federated smoke runs (not RETFound)."""

    def __init__(self, in_dim: int, n_labels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, int(n_labels)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def _local_train_one_client(
    model: TinyMLP,
    loader: DataLoader,
    *,
    cfg: FedSmokeConfig,
    global_sd: OrderedDict[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], float, int]:
    device = torch.device(cfg.device)
    model.train()
    model.load_state_dict(global_sd)

    opt = torch.optim.SGD(model.parameters(), lr=float(cfg.lr))
    loss_fn = nn.BCEWithLogitsLoss()

    global_vec = torch.nn.utils.parameters_to_vector(
        [p.detach().clone().to(device) for p in model.parameters()],
    )

    total_loss = 0.0
    n_batches = 0
    for _ in range(int(cfg.local_epochs)):
        for x, y, _meta in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            if float(cfg.fedprox_mu) > 0.0:
                cur_vec = torch.nn.utils.parameters_to_vector(list(model.parameters()))
                prox = float(cfg.fedprox_mu) * (cur_vec - global_vec).pow(2).sum()
                loss = loss + 0.5 * prox
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu().item())
            n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    sd_out = OrderedDict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    return sd_out, mean_loss, n_batches


def run_multilabel_fed_smoke(
    *,
    labels_csv: Path,
    images_dir: Path,
    split_json: Path,
    image_size: tuple[int, int] = (32, 32),
    cfg: FedSmokeConfig | None = None,
    noise_protocol_yaml: Path | None = None,
    label_noise_seed: int = 0,
) -> dict[str, Any]:
    """Run a tiny FedAvg/FedProx smoke loop over client splits (Torch required)."""

    cfg = cfg or FedSmokeConfig()

    label_noise_p_flip: float | None = None
    noise_meta: dict[str, Any] | None = None
    if noise_protocol_yaml is not None:
        np_path = Path(noise_protocol_yaml)
        raw = load_noise_protocol_yaml(np_path)
        proto = parse_noise_protocol_v1(raw)
        label_noise_p_flip = float(proto.symmetric_flip_p_flip)
        noise_meta = {
            "path": str(np_path.as_posix()),
            "symmetric_flip_p_flip": label_noise_p_flip,
        }

    ds = RFMiDTorchDataset(
        labels_csv=labels_csv,
        images_dir=images_dir,
        image_size=image_size,
        label_noise_p_flip=label_noise_p_flip,
        label_noise_seed=int(label_noise_seed),
    )
    if len(ds) == 0:
        raise ValueError("Dataset is empty")

    # Infer label dim from one sample
    _x0, y0, _m0 = ds[0]
    n_labels = int(y0.shape[0])
    in_dim = int(_x0.numel())

    split = read_split_json(Path(split_json))
    clients: dict[str, list[str]] = split["clients"]
    client_keys = sorted(clients.keys(), key=lambda k: int(k))

    global_model = TinyMLP(in_dim=in_dim, n_labels=n_labels)
    global_sd = OrderedDict({k: v.detach().cpu() for k, v in global_model.state_dict().items()})

    metrics: dict[str, Any] = {
        "rounds": [],
        "comm_bytes_upload_per_round": [],
        "mean_train_loss_clients": [],
    }
    if noise_meta is not None:
        metrics["noise_protocol"] = noise_meta

    for r in range(int(cfg.rounds)):
        local_sds: list[OrderedDict[str, torch.Tensor]] = []
        weights: list[float] = []
        losses: list[float] = []

        upload_bytes_round = 0
        for ck in client_keys:
            ids = clients[ck]
            idxs = ds.indices_for_ids(list(ids))
            if not idxs:
                continue
            subset = Subset(ds, idxs)
            loader = DataLoader(subset, batch_size=int(cfg.batch_size), shuffle=True)

            local_sd, mean_loss, _n = _local_train_one_client(
                global_model,
                loader,
                cfg=cfg,
                global_sd=global_sd,
            )
            local_sds.append(local_sd)
            weights.append(float(len(idxs)))
            losses.append(mean_loss)
            upload_bytes_round += state_dict_nbytes(local_sd)

        if not local_sds:
            raise ValueError("No clients had data; check split JSON vs dataset IDs")

        global_sd = fedavg_state_dict(local_sds, weights)
        global_model.load_state_dict(global_sd)

        metrics["rounds"].append(r)
        metrics["comm_bytes_upload_per_round"].append(upload_bytes_round)
        metrics["mean_train_loss_clients"].append(float(sum(losses) / len(losses)))

    metrics["final_state_dict_keys"] = list(global_sd.keys())
    metrics["total_upload_bytes"] = int(sum(metrics["comm_bytes_upload_per_round"]))
    return metrics
