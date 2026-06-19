from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_minimal_experiment_writes_json(tmp_path: Path) -> None:
    import importlib

    mod = importlib.import_module("fed_agent.tools.minimal_experiment")
    out = tmp_path / "m.json"
    snap = tmp_path / "snap.md"
    rc = mod.main(
        [
            "--workdir",
            str(tmp_path / "fixture"),
            "--out_json",
            str(out),
            "--write_docs_snapshot",
            str(snap),
            "--rounds",
            "1",
            "--fedprox_mu",
            "0.0",
        ],
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["total_upload_bytes"] > 0
    assert "Minimal synthetic" in snap.read_text(encoding="utf-8")
