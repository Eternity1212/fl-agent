from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fed_agent.noise.protocol import apply_symmetric_label_noise


def test_symmetric_flip_changes_only_positives() -> None:
    rng = np.random.default_rng(0)
    y = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y2 = apply_symmetric_label_noise(y, rng=rng, p_flip=1.0)
    # With p_flip=1, positives become 0
    assert y2[0, 0] == 0.0
    assert y2[1, 1] == 0.0
    assert y2[0, 1] == 0.0
    assert y2[1, 0] == 0.0


def test_symmetric_flip_zero_prob_noop() -> None:
    rng = np.random.default_rng(1)
    y = np.array([[1.0, 1.0]], dtype=np.float32)
    y2 = apply_symmetric_label_noise(y, rng=rng, p_flip=0.0)
    assert np.allclose(y, y2)


def test_load_noise_protocol_yaml(tmp_path: Path) -> None:
    from fed_agent.noise.protocol import load_noise_protocol_yaml

    p = tmp_path / "np.yaml"
    yaml_body = (
        'version: "noise_protocol@v1"\n'
        "symmetric_flip_on_positives:\n"
        "  p_flip: 0.1\n"
    )
    p.write_text(yaml_body, encoding="utf-8")
    d = load_noise_protocol_yaml(p)
    assert d["version"] == "noise_protocol@v1"
    assert float(d["symmetric_flip_on_positives"]["p_flip"]) == 0.1


def test_parse_noise_protocol_v1_ok() -> None:
    from fed_agent.noise.protocol import parse_noise_protocol_v1

    proto = parse_noise_protocol_v1(
        {"version": "noise_protocol@v1", "symmetric_flip_on_positives": {"p_flip": 0.25}},
    )
    assert proto.symmetric_flip_p_flip == 0.25


def test_parse_noise_protocol_v1_omitted_sym_defaults_zero() -> None:
    from fed_agent.noise.protocol import parse_noise_protocol_v1

    proto = parse_noise_protocol_v1({"version": "noise_protocol@v1"})
    assert proto.symmetric_flip_p_flip == 0.0


def test_parse_noise_protocol_v1_bad_version() -> None:
    from fed_agent.noise.protocol import parse_noise_protocol_v1

    with pytest.raises(ValueError, match="expected version"):
        parse_noise_protocol_v1({"version": "x"})
