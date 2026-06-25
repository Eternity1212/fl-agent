# Optimized matrix artifacts and history

## Branch

- Branch: `results/retfound-optimized-summary`

## Local raw artifacts (gitignored)

These are the authoritative raw outputs for the completed optimized matrix run:

- `runs/paper_matrix/retfound_full_optimized_resume/summary.json`
- `runs/paper_matrix/retfound_full_optimized_resume/*.json`

The repository does **not** commit these raw files because `runs/` is gitignored in `.gitignore`.

## Committed reproducibility artifacts

The following committed files capture the experiment configuration, execution changes, and written results:

- `src/fed_agent/train/paper_runner.py`
- `configs/paper_matrix_gpu_full_optimized.yaml`
- `configs/paper_matrix_gpu_full_optimized_robust.yaml`
- `docs/results/paper_matrix_retfound_optimized_summary.md`
- `docs/results/paper_matrix_retfound_optimized_results_section.md`
- `docs/results/paper_matrix_retfound_optimized_mean_std.md`
- `docs/results/paper_matrix_retfound_optimized_artifacts.md`

## Experiment history captured in git

This branch preserves the changes needed to:

1. fix the federated device mismatch issue,
2. add the optimized matrix configs,
3. enable the optimized execution path,
4. document the final completed run and its main findings.

For the full per-run numerical history, consult the local `runs/paper_matrix/retfound_full_optimized_resume/` directory together with the written summaries committed on this branch.
