from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def test_summarize_fed_smoke_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "m.json"
    out.write_text(
        json.dumps(
            {
                "noise_protocol": {
                    "path": "/tmp/np.yaml",
                    "symmetric_flip_p_flip": 0.1,
                },
                "total_upload_bytes": 1200,
                "mean_train_loss_clients": [0.9, 0.8],
                "comm_bytes_upload_per_round": [600, 600],
                "final_state_dict_keys": ["a", "b"],
            },
        ),
        encoding="utf-8",
    )
    mod = importlib.import_module("fed_agent.tools.summarize_fed_smoke")
    monkeypatch.setattr("sys.argv", ["summarize_fed_smoke", str(out)])
    assert mod.main(None) == 0
    captured = capsys.readouterr().out
    assert "total_upload_bytes: 1200" in captured
    assert "noise_protocol.path:" in captured
