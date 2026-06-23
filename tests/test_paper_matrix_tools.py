from __future__ import annotations

from fed_agent.tools.run_paper_matrix import _cfg_from_row
from fed_agent.tools.summarize_paper_matrix import build_markdown


def test_cfg_from_row_merges_defaults() -> None:
    cfg = _cfg_from_row(
        {"backbone": "mlp", "image_size": [8, 8], "batch_size": 2},
        {"name": "x", "method": "centralized", "loss": "bce"},
    )
    assert cfg.method == "centralized"
    assert cfg.backbone == "mlp"
    assert cfg.image_size == (8, 8)


def test_build_markdown_marks_retfound_flag() -> None:
    md = build_markdown(
        [
            {
                "name": "x",
                "method": "fedavg",
                "loss": "bce",
                "positive_dropout": 0.0,
                "train_label_noise": 0.0,
                "best_macro_f1_present": 0.1,
                "best_micro_f1": 0.2,
                "macro_auroc": 0.5,
                "total_upload_bytes": 10,
                "is_retfound": False,
            },
        ],
    )
    assert "RETFound=False" in md
    assert "| x | fedavg |" in md
