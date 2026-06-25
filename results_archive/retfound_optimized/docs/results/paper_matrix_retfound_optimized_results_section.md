# Results — RETFound + LoRA Federated RFMiD (optimized matrix)

This section summarizes the completed optimized matrix run stored locally under:

- Raw per-run JSONs: `runs/paper_matrix/retfound_full_optimized_resume/*.json`
- Combined raw summary: `runs/paper_matrix/retfound_full_optimized_resume/summary.json`

Because `runs/` is gitignored, the repository stores a curated written summary rather than the raw JSON payloads.

## Experimental setting

- Backbone: **RETFound ViT-Large** with **LoRA**
- Dataset: **RFMiD**
- Main comparison methods: Centralized, Local-only, FedAvg, FedProx, Robust-FedProx
- Clean comparison seeds: **0, 1, 2**
- Additional ablations: label noise, Dirichlet non-IID, LoRA rank, FedProx `mu`

## Main clean-setting comparison

### Centralized remains the upper bound

Across all available seeds, centralized RETFound + LoRA remains the strongest setting. This confirms that the pretrained backbone and PEFT adaptation are effective for the target multi-label task.

### Local-only is consistently weaker

The local-only runs are stable but substantially below centralized performance, indicating that cross-client collaboration is useful in this benchmark.

### FedAvg is the strongest federated baseline

Across the completed clean 3-seed comparison, **FedAvg** consistently outperforms both **FedProx** and **Robust-FedProx**.

Representative trend:

- `FedAvg` achieves roughly **0.626–0.631 micro-F1** and **0.218–0.249 macro-F1-present**.
- `FedProx` stays near **0.534–0.537 micro-F1** and **0.159–0.165 macro-F1-present**.
- `Robust-FedProx` stays near **0.530–0.552 micro-F1** and **0.156–0.166 macro-F1-present**.

This result is qualitatively stable across seeds and does not support a clean-setting claim that FedProx or Robust-FedProx surpasses FedAvg.

## Noise and non-IID ablations

### Label noise

For the completed noise ablations, the available results do not show Robust-FedProx overtaking FedAvg under label noise. In particular, at `p_flip = 0.2`, the FedAvg run remains clearly stronger than the Robust-FedProx counterpart.

### Dirichlet non-IID

For the completed Dirichlet `alpha=0.1` case, the currently available comparison also favors FedAvg over Robust-FedProx. Thus, the expected "robust method wins in harder conditions" story is not yet supported by the completed matrix.

## Interpretation

### Supported claims

1. **RETFound + LoRA works** for RFMiD.
2. **Federated learning is useful** because it improves over local-only training.
3. **FedAvg is the strongest stable federated baseline** in the completed optimized matrix.
4. The optimized execution path reproduced the same qualitative conclusions while allowing the full matrix to complete successfully.

### Unsupported claims

1. Clean-setting superiority of **FedProx** over FedAvg.
2. Clean-setting superiority of **Robust-FedProx** over FedAvg.
3. A strong headline claim that the current robust regularization already provides the best federated solution in this RETFound + LoRA regime.

## Recommended framing

A defensible framing of the completed experiments is:

> We establish a RETFound + LoRA federated benchmark on RFMiD and show that FedAvg is a strong and stable baseline. In contrast, straightforward FedProx and Robust-FedProx variants do not automatically improve over FedAvg in this foundation-model + PEFT setting, suggesting that more suitable robust federated objectives remain an open problem.
