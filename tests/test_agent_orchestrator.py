"""Unit tests for the adaptive orchestration agent's weighting policy."""

from __future__ import annotations

from fed_agent.agent.orchestrator import decide_client_mu, decide_weights


def test_equal_scores_reduce_to_size_weights():
    # All clients equally good -> size-proportional (FedAvg) weights.
    d = decide_weights(probe_scores=[-0.5, -0.5, -0.5, -0.5], sizes=[40, 40, 20, 20])
    total = sum(d.weights)
    assert abs(total - 1.0) < 1e-9
    # client 0 has twice the data of client 2 -> twice the weight.
    assert abs(d.weights[0] / d.weights[2] - 2.0) < 1e-6


def test_noisy_clients_are_suppressed_clean_stay_balanced():
    # Clients 0,1 clean (high probe) ; 2,3 noisy (low probe).
    d = decide_weights(
        probe_scores=[-0.55, -0.53, -0.70, -0.71],
        sizes=[40, 40, 40, 40],
        tau=0.03,
    )
    clean = d.weights[0] + d.weights[1]
    noisy = d.weights[2] + d.weights[3]
    assert clean > 0.8, f"clean mass should dominate, got {clean}"
    assert noisy < 0.2, f"noisy mass should be suppressed, got {noisy}"
    # Two equally-clean clients must stay balanced (no collapse onto one).
    assert abs(d.weights[0] - d.weights[1]) < 0.1


def test_empty_input():
    d = decide_weights(probe_scores=[], sizes=[])
    assert d.weights == []


def test_adaptive_mu_zero_when_clean():
    # Equal scores -> nobody is below the pack -> mu ~ 0 (reduces to FedAvg).
    mus = decide_client_mu(probe_scores=[-0.5, -0.5, -0.5, -0.5], mu_max=0.1, tau=0.05)
    assert all(m == 0.0 for m in mus)


def test_adaptive_mu_strong_for_noisy_clients():
    # Clients 0,1 clean (high) ; 2,3 noisy (low) -> noisy get strong mu, clean 0.
    mus = decide_client_mu(
        probe_scores=[-0.50, -0.52, -0.70, -0.72], mu_max=0.1, tau=0.05
    )
    assert mus[0] == 0.0 and mus[1] == 0.0
    assert mus[2] > 0.05 and mus[3] > 0.05
