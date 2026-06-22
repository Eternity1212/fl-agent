from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fed_agent.data.rfmid import load_rfmid_label_table
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
    seed: int = 0
    model: str = "mlp"
    loss: str = "bce"
    threshold: float = 0.5


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


class TinyCNN(nn.Module):
    """Small CNN for RFMiD subset smoke runs (still not RETFound)."""

    def __init__(self, n_labels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, int(n_labels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.features(x).flatten(1)
        return self.head(z)


def _make_model(*, model: str, in_dim: int, n_labels: int) -> nn.Module:
    if model == "mlp":
        return TinyMLP(in_dim=in_dim, n_labels=n_labels)
    if model == "tiny_cnn":
        return TinyCNN(n_labels=n_labels)
    raise ValueError(f"Unknown model: {model!r}")


def _pos_weight_from_labels(labels_csv: Path) -> torch.Tensor:
    _ids, _names, y = load_rfmid_label_table(labels_csv)
    pos = torch.tensor(y.sum(axis=0), dtype=torch.float32)
    neg = torch.tensor(float(y.shape[0]), dtype=torch.float32) - pos
    # Clip to keep the smoke run numerically tame on rare labels.
    w = torch.where(pos > 0, neg / torch.clamp(pos, min=1.0), torch.ones_like(pos))
    return torch.clamp(w, min=1.0, max=20.0)


def _f1_scores(y_true: torch.Tensor, y_prob: torch.Tensor, *, threshold: float) -> dict[str, float]:
    yt = (y_true.cpu().numpy() > 0.5).astype("int64")
    yp = (y_prob.cpu().numpy() >= float(threshold)).astype("int64")

    tp = (yt * yp).sum()
    fp = ((1 - yt) * yp).sum()
    fn = (yt * (1 - yp)).sum()
    micro = float((2 * tp) / max((2 * tp + fp + fn), 1))

    per_label: list[float] = []
    for j in range(int(yt.shape[1])):
        yt_j = yt[:, j]
        yp_j = yp[:, j]
        if yt_j.sum() == 0:
            continue
        tp_j = int((yt_j * yp_j).sum())
        fp_j = int(((1 - yt_j) * yp_j).sum())
        fn_j = int((yt_j * (1 - yp_j)).sum())
        per_label.append(float((2 * tp_j) / max((2 * tp_j + fp_j + fn_j), 1)))

    macro_present = float(sum(per_label) / len(per_label)) if per_label else 0.0
    return {"micro_f1": micro, "macro_f1_present": macro_present}


def _best_threshold_scores(y_true: torch.Tensor, y_prob: torch.Tensor) -> dict[str, float]:
    best_t = 0.5
    best_macro = -1.0
    best_micro = 0.0
    for t in [i / 100.0 for i in range(5, 96, 5)]:
        scores = _f1_scores(y_true, y_prob, threshold=t)
        if scores["macro_f1_present"] > best_macro:
            best_t = t
            best_macro = scores["macro_f1_present"]
            best_micro = scores["micro_f1"]
    return {
        "best_threshold": float(best_t),
        "best_macro_f1_present": float(best_macro),
        "best_micro_f1": float(best_micro),
    }


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    cfg: FedSmokeConfig,
    pos_weight: torch.Tensor | None,
) -> dict[str, float]:
    device = torch.device(cfg.device)
    model.eval()
    pos_w = pos_weight.to(device) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    losses: list[float] = []
    y_all: list[torch.Tensor] = []
    p_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y, _meta in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            losses.append(float(loss_fn(logits, y).detach().cpu().item()))
            y_all.append(y.detach().cpu())
            p_all.append(torch.sigmoid(logits).detach().cpu())
    if not losses:
        return {"loss": float("nan"), "micro_f1": 0.0, "macro_f1_present": 0.0}
    y_cat = torch.cat(y_all, dim=0)
    p_cat = torch.cat(p_all, dim=0)
    scores = _f1_scores(y_cat, p_cat, threshold=float(cfg.threshold))
    best = _best_threshold_scores(y_cat, p_cat)
    return {"loss": float(sum(losses) / len(losses)), **scores, **best}


def _local_train_one_client(
    model: nn.Module,
    loader: DataLoader,
    *,
    cfg: FedSmokeConfig,
    global_sd: OrderedDict[str, torch.Tensor],
    pos_weight: torch.Tensor | None,
) -> tuple[OrderedDict[str, torch.Tensor], float, int]:
    device = torch.device(cfg.device)
    model.train()
    model.load_state_dict(global_sd)

    opt = torch.optim.SGD(model.parameters(), lr=float(cfg.lr))
    pos_w = pos_weight.to(device) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

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
    eval_labels_csv: Path | None = None,
    eval_images_dir: Path | None = None,
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

    torch.manual_seed(int(cfg.seed))
    global_model = _make_model(model=str(cfg.model), in_dim=in_dim, n_labels=n_labels)
    global_sd = OrderedDict({k: v.detach().cpu() for k, v in global_model.state_dict().items()})
    pos_weight = _pos_weight_from_labels(labels_csv) if str(cfg.loss) == "balanced_bce" else None

    eval_loader: DataLoader | None = None
    if eval_labels_csv is not None and eval_images_dir is not None:
        eval_ds = RFMiDTorchDataset(
            labels_csv=eval_labels_csv,
            images_dir=eval_images_dir,
            image_size=image_size,
        )
        eval_loader = DataLoader(eval_ds, batch_size=int(cfg.batch_size), shuffle=False)

    metrics: dict[str, Any] = {
        "rounds": [],
        "comm_bytes_upload_per_round": [],
        "mean_train_loss_clients": [],
    }
    if pos_weight is not None:
        metrics["loss"] = {"name": "balanced_bce", "pos_weight_max": float(pos_weight.max().item())}
    else:
        metrics["loss"] = {"name": "bce"}
    metrics["model"] = str(cfg.model)
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
            generator = torch.Generator()
            generator.manual_seed(int(cfg.seed) + r * 10_000 + int(ck))
            loader = DataLoader(
                subset,
                batch_size=int(cfg.batch_size),
                shuffle=True,
                generator=generator,
            )

            local_sd, mean_loss, _n = _local_train_one_client(
                global_model,
                loader,
                cfg=cfg,
                global_sd=global_sd,
                pos_weight=pos_weight,
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
        if eval_loader is not None:
            ev = _evaluate(global_model, eval_loader, cfg=cfg, pos_weight=pos_weight)
            metrics.setdefault("eval_loss", []).append(ev["loss"])
            metrics.setdefault("eval_micro_f1", []).append(ev["micro_f1"])
            metrics.setdefault("eval_macro_f1_present", []).append(ev["macro_f1_present"])
            metrics.setdefault("eval_best_threshold", []).append(ev["best_threshold"])
            metrics.setdefault("eval_best_micro_f1", []).append(ev["best_micro_f1"])
            metrics.setdefault("eval_best_macro_f1_present", []).append(ev["best_macro_f1_present"])

    metrics["final_state_dict_keys"] = list(global_sd.keys())
    metrics["total_upload_bytes"] = int(sum(metrics["comm_bytes_upload_per_round"]))
    return metrics
