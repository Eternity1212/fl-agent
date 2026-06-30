# Agent 全矩阵最新结果索引 (截至 2026-06-30)

> 核心指标用 **macro_auroc / macro_ap**(阈值无关、多标签稳健)。best_micro_f1 受阈值选择影响大,仅作参考。

| 文件 | method | agg | seed | macro_auroc | macro_ap | best_micro_f1 |
|---|---|---|---|---|---|---|
| agent/clean_agent_s0.json | agent_fed | agent | 0 | 0.8265 | 0.2905 | 0.6936 |
| agent/clean_fedavg_s0.json | agent_fed | size | 0 | 0.8249 | 0.2796 | 0.6599 |
| agent/het02_agent_s0.json | agent_fed | agent | 0 | 0.7812 | 0.2663 | 0.6590 |
| agent/het02_fedavg_s0.json | agent_fed | size | 0 | 0.7211 | 0.2426 | 0.3035 |
| agent/het02_robust_s0.json | agent_fed | size | 0 | 0.7302 | 0.1914 | 0.5956 |
| agent/het04_agentmu_s0.json | agent_fed | agent | 0 | 0.7973 | 0.2720 | 0.7057 |
| agent/het04_agentmu_s1.json | agent_fed | agent | 1 | 0.8309 | 0.2611 | 0.6518 |
| agent/het04_agentmu_s2.json | agent_fed | agent | 2 | 0.7874 | 0.2558 | 0.6034 |
| agent/het04_dir_agentmu_s0.json | agent_fed | agent | 0 | 0.6040 | 0.0746 | 0.0000 |
| agent/het04_muonly_s0.json | agent_fed | size | 0 | 0.7669 | 0.2287 | 0.4790 |
| agent_stage1/het04_agent_s0.json | agent_fed | agent | 0 | 0.7908 | 0.2705 | 0.7007 |
| agent_stage1/het04_agent_s1.json | agent_fed | agent | 1 | 0.8102 | 0.2831 | 0.6752 |
| agent_stage1/het04_agent_s2.json | agent_fed | agent | 2 | 0.8120 | 0.2715 | 0.6869 |
| agent_stage1/het04_dir_agent_s0.json | agent_fed | agent | 0 | 0.5330 | 0.0640 | 0.0000 |
| agent_stage1/het04_dir_fedavg_s0.json | agent_fed | size | 0 | 0.5876 | 0.0765 | 0.0784 |
| agent_stage1/het04_fedavg_s0.json | agent_fed | size | 0 | 0.7374 | 0.2502 | 0.1469 |
| agent_stage1/het04_fedavg_s1.json | agent_fed | size | 1 | 0.7377 | 0.2239 | 0.3686 |
| agent_stage1/het04_fedavg_s2.json | agent_fed | size | 2 | 0.7389 | 0.2491 | 0.2608 |
| agent_stage1/het04_robust_s0.json | agent_fed | size | 0 | 0.6669 | 0.1626 | 0.0742 |
| agent_stage1/het04_robust_s1.json | agent_fed | size | 1 | 0.6762 | 0.1714 | 0.0561 |
| agent_stage1/het04_robust_s2.json | agent_fed | size | 2 | 0.6637 | 0.1663 | 0.0841 |
