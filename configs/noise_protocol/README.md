# Noise protocol configs

`example_v1.yaml` is consumed by `fed_agent.noise.protocol` and can be passed to federated smoke:

```bash
python3 -m fed_agent.tools.run_fed_smoke ... \
  --noise_protocol_yaml configs/noise_protocol/example_v1.yaml
```

Symmetric flip on positives is applied inside `RFMiDTorchDataset` (deterministic per index + `--label_noise_seed`).

Next steps:

- Asymmetric / class-conditional noise modes (v0.4+)
- FedDiv-style baselines and ablations
