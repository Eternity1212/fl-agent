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
    def row(name: str, split: str, loss: str, macro: float, noise_p=None) -> dict[str, object]:
        return {
            "name": name,
            "split": split,
            "fedprox_mu": 0.05 if "fedprox" in name else 0.0,
            "noise_p": noise_p,
            "loss": loss,
            "model": "tiny_cnn",
            "final_train_loss": 1.0,
            "start_train_loss": 1.2,
            "final_eval_loss": 0.8,
            "final_eval_micro_f1": macro,
            "final_eval_macro_f1_present": macro,
            "best_eval_threshold": 0.3,
            "best_eval_micro_f1": macro,
            "best_eval_macro_f1_present": macro,
            "total_upload_bytes": 10,
        }

    rows = [
        row("iid_fedavg_bce_clean", "iid", "bce", 0.10),
        row("iid_fedavg_balanced_clean", "iid", "balanced_bce", 0.20),
        row("iid_fedprox_balanced_clean", "iid", "balanced_bce", 0.21),
        row("iid_fedavg_balanced_noise_p01", "iid", "balanced_bce", 0.18, 0.1),
        row("iid_fedprox_balanced_noise_p01", "iid", "balanced_bce", 0.19, 0.1),
        row("dirichlet_a0p5_fedavg_bce_clean", "dirichlet_a0p5", "bce", 0.09),
        row("dirichlet_a0p5_fedavg_balanced_clean", "dirichlet_a0p5", "balanced_bce", 0.18),
        row("dirichlet_a0p5_fedprox_balanced_clean", "dirichlet_a0p5", "balanced_bce", 0.19),
        row(
            "dirichlet_a0p5_fedavg_balanced_noise_p01",
            "dirichlet_a0p5",
            "balanced_bce",
            0.17,
            0.1,
        ),
        row(
            "dirichlet_a0p5_fedprox_balanced_noise_p01",
            "dirichlet_a0p5",
            "balanced_bce",
            0.18,
            0.1,
        ),
    ]
    md = _report(rows)
    assert "Method comparison" in md
    assert "Robustness / noise ablation" in md
