from __future__ import annotations

from fed_agent.tools.export_hf_rfmid_subset import _select_diverse_rows
from fed_agent.tools.run_rfmid_smoke_matrix import _report


def test_select_diverse_rows_round_robin_by_disease() -> None:
    rows = [
        {"ID": "1", "Disease_Risk": "1", "A": "1", "B": "0"},
        {"ID": "2", "Disease_Risk": "1", "A": "1", "B": "0"},
        {"ID": "3", "Disease_Risk": "1", "A": "0", "B": "1"},
        {"ID": "4", "Disease_Risk": "1", "A": "0", "B": "1"},
    ]
    out = _select_diverse_rows(
        rows,
        id_key="ID",
        label_keys=["Disease_Risk", "A", "B"],
        max_samples=2,
    )
    assert [r["ID"] for r in out] == ["1", "3"]


def test_rfmid_smoke_report_contains_method_sections() -> None:
    rows = [
        {
            "name": "iid_fedavg_clean",
            "split": "iid",
            "fedprox_mu": 0.0,
            "noise_p": None,
            "final_train_loss": 1.0,
            "start_train_loss": 1.2,
            "total_upload_bytes": 10,
        },
        {
            "name": "iid_fedprox_clean",
            "split": "iid",
            "fedprox_mu": 0.05,
            "noise_p": None,
            "final_train_loss": 0.9,
            "start_train_loss": 1.2,
            "total_upload_bytes": 10,
        },
        {
            "name": "iid_fedavg_noise_p01",
            "split": "iid",
            "fedprox_mu": 0.0,
            "noise_p": 0.1,
            "final_train_loss": 1.1,
            "start_train_loss": 1.2,
            "total_upload_bytes": 10,
        },
        {
            "name": "iid_fedprox_noise_p01",
            "split": "iid",
            "fedprox_mu": 0.05,
            "noise_p": 0.1,
            "final_train_loss": 1.0,
            "start_train_loss": 1.2,
            "total_upload_bytes": 10,
        },
        {
            "name": "dirichlet_a0p5_fedavg_clean",
            "split": "dirichlet_a0p5",
            "fedprox_mu": 0.0,
            "noise_p": None,
            "final_train_loss": 1.2,
            "start_train_loss": 1.3,
            "total_upload_bytes": 10,
        },
        {
            "name": "dirichlet_a0p5_fedprox_clean",
            "split": "dirichlet_a0p5",
            "fedprox_mu": 0.05,
            "noise_p": None,
            "final_train_loss": 1.1,
            "start_train_loss": 1.3,
            "total_upload_bytes": 10,
        },
        {
            "name": "dirichlet_a0p5_fedavg_noise_p01",
            "split": "dirichlet_a0p5",
            "fedprox_mu": 0.0,
            "noise_p": 0.1,
            "final_train_loss": 1.3,
            "start_train_loss": 1.3,
            "total_upload_bytes": 10,
        },
        {
            "name": "dirichlet_a0p5_fedprox_noise_p01",
            "split": "dirichlet_a0p5",
            "fedprox_mu": 0.05,
            "noise_p": 0.1,
            "final_train_loss": 1.2,
            "start_train_loss": 1.3,
            "total_upload_bytes": 10,
        },
    ]
    md = _report(rows)
    assert "Method comparison" in md
    assert "Robustness / noise ablation" in md
