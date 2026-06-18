from __future__ import annotations

from pathlib import Path

from fed_agent.tools.build_splits import main


def _tiny_csv(tmp_path: Path) -> Path:
    p = tmp_path / "labels.csv"
    p.write_text(
        "\n".join(
            [
                "ImageID,A,B",
                "s0,1,0",
                "s1,1,0",
                "s2,0,1",
                "s3,0,1",
                "s4,1,1",
                "s5,0,0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return p


def test_build_splits_writes_json(tmp_path: Path) -> None:
    csv_path = _tiny_csv(tmp_path)
    out_dir = tmp_path / "out"
    rc = main(
        [
            "--labels_csv",
            str(csv_path),
            "--out_dir",
            str(out_dir),
            "--n_clients",
            "3",
            "--seed",
            "7",
            "--alphas",
            "0.5",
        ],
    )
    assert rc == 0
    files = sorted(out_dir.glob("*.json"))
    assert len(files) == 3  # iid + domain_hash + one dirichlet
