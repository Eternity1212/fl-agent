# Federated simulation (v0.3+)

This repo includes a **tiny Torch smoke simulator** intended to validate wiring:

- **Split JSON** (`fl_agent.split.v1`) → per-client `Subset` of `RFMiDTorchDataset`
- **FedAvg** on full model `state_dict` (not LoRA-only yet)
- Optional **FedProx** proximal term (`FedSmokeConfig.fedprox_mu`)
- **Upload bytes / round** = sum of uploaded `state_dict` tensor bytes per round
- Optional **label noise** from `noise_protocol@v1` YAML (symmetric flip on positives), wired into `RFMiDTorchDataset` and echoed in metrics as `noise_protocol`

## CLI

```bash
python3 -m fed_agent.tools.run_fed_smoke \
  --labels_csv path/to/labels.csv \
  --images_dir path/to/images \
  --split_json path/to/split.json \
  --rounds 2 \
  --fedprox_mu 0.0 \
  --noise_protocol_yaml configs/noise_protocol/example_v1.yaml \
  --label_noise_seed 0 \
  --out_json runs/metrics.json
```

Torch is required for this path (`pip install -e ".[torch]"`).

## Using the JSON output

```bash
python3 -m fed_agent.tools.summarize_fed_smoke runs/metrics.json
```

Example field layout: [examples/fed_smoke_metrics.example.json](examples/fed_smoke_metrics.example.json).

## Next milestones

- Replace `TinyMLP` with **RETFound + LoRA** and count **uploaded adapter bytes only**
- Richer noise modes (asymmetric / class-conditional) + FedDiv-style baselines
- Add **GlobalAgent / RuleAgent** scheduling on top of the same loop
