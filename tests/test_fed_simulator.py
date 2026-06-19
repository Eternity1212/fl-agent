from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from fed_agent.fed.simulator import FedSmokeConfig, run_multilabel_fed_smoke
from fed_agent.splits.partition import split_payload, write_split_json


def _write_tiny_fed_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    for sid in ["s0", "s1", "s2", "s3"]:
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img_dir / f"{sid}.png")

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ImageID,A,B",
                "s0,1,0",
                "s1,1,0",
                "s2,0,1",
                "s3,0,1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from fed_agent.data.rfmid import load_rfmid_label_table

    image_ids, label_names, y = load_rfmid_label_table(csv_path)
    y_lookup = {sid: y[i] for i, sid in enumerate(image_ids)}
    clients = {"0": ["s0", "s1"], "1": ["s2", "s3"]}
    payload = split_payload(
        split="iid_test",
        n_clients=2,
        seed=0,
        alpha=None,
        label_names=label_names,
        clients=clients,
        y_lookup=y_lookup,
    )
    split_path = tmp_path / "split.json"
    write_split_json(split_path, payload)
    return csv_path, img_dir, split_path


def test_run_multilabel_fed_smoke_runs(tmp_path: Path) -> None:
    labels_csv, images_dir, split_json = _write_tiny_fed_fixture(tmp_path)
    cfg = FedSmokeConfig(
        rounds=2,
        local_epochs=1,
        batch_size=2,
        lr=0.1,
        fedprox_mu=0.01,
        device="cpu",
    )
    metrics = run_multilabel_fed_smoke(
        labels_csv=labels_csv,
        images_dir=images_dir,
        split_json=split_json,
        image_size=(8, 8),
        cfg=cfg,
    )
    assert metrics["total_upload_bytes"] > 0
    assert len(metrics["comm_bytes_upload_per_round"]) == 2


def test_run_fed_smoke_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    labels_csv, images_dir, split_json = _write_tiny_fed_fixture(tmp_path)
    out = tmp_path / "metrics.json"

    run_fed_smoke = importlib.import_module("fed_agent.tools.run_fed_smoke")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_fed_smoke",
            "--labels_csv",
            str(labels_csv),
            "--images_dir",
            str(images_dir),
            "--split_json",
            str(split_json),
            "--rounds",
            "1",
            "--image_size",
            "8",
            "8",
            "--out_json",
            str(out),
        ],
    )
    rc = run_fed_smoke.main(None)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "total_upload_bytes" in payload


def test_run_multilabel_fed_smoke_with_noise_yaml(tmp_path: Path) -> None:
    labels_csv, images_dir, split_json = _write_tiny_fed_fixture(tmp_path)
    np_path = tmp_path / "np.yaml"
    np_path.write_text(
        "\n".join(
            [
                'version: "noise_protocol@v1"',
                "symmetric_flip_on_positives:",
                "  p_flip: 0.0",
                "",
            ],
        ),
        encoding="utf-8",
    )
    cfg = FedSmokeConfig(
        rounds=1,
        local_epochs=1,
        batch_size=2,
        lr=0.1,
        fedprox_mu=0.0,
        device="cpu",
    )
    metrics = run_multilabel_fed_smoke(
        labels_csv=labels_csv,
        images_dir=images_dir,
        split_json=split_json,
        image_size=(8, 8),
        cfg=cfg,
        noise_protocol_yaml=np_path,
        label_noise_seed=7,
    )
    assert metrics["noise_protocol"]["path"] == str(np_path.as_posix())
    assert metrics["noise_protocol"]["symmetric_flip_p_flip"] == 0.0


def test_run_fed_smoke_cli_with_noise_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    labels_csv, images_dir, split_json = _write_tiny_fed_fixture(tmp_path)
    np_path = tmp_path / "np.yaml"
    np_path.write_text(
        "\n".join(
            [
                'version: "noise_protocol@v1"',
                "symmetric_flip_on_positives:",
                "  p_flip: 0.0",
                "",
            ],
        ),
        encoding="utf-8",
    )
    out = tmp_path / "metrics.json"
    run_fed_smoke = importlib.import_module("fed_agent.tools.run_fed_smoke")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_fed_smoke",
            "--labels_csv",
            str(labels_csv),
            "--images_dir",
            str(images_dir),
            "--split_json",
            str(split_json),
            "--rounds",
            "1",
            "--image_size",
            "8",
            "8",
            "--noise_protocol_yaml",
            str(np_path),
            "--label_noise_seed",
            "9",
            "--out_json",
            str(out),
        ],
    )
    assert run_fed_smoke.main(None) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["noise_protocol"]["symmetric_flip_p_flip"] == 0.0
