from __future__ import annotations

import csv
from pathlib import Path

from fed_agent.tools.project_label_columns import project_label_columns


def test_project_label_columns(tmp_path: Path) -> None:
    src = tmp_path / "labels.csv"
    src.write_text("ImageID,A,B,C\n1,1,0,1\n2,0,1,0\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    project_label_columns(labels_csv=src, out_csv=out, labels=["C", "A"])
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0] == {"ImageID": "1", "C": "1", "A": "1"}
    assert rows[1] == {"ImageID": "2", "C": "0", "A": "0"}
