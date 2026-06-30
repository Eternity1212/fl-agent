# Agent 全矩阵最新结果存档 (2026-06-30)

本目录是**当前最权威、最新的一份 agent 实验结果集合**,合并了两批 GPU 产出:

- `agent_stage1/` —— 2026-06-26 那批(stage1 决定性对照):het04 的 fedavg/robust/agent 各 3 个 seed,以及非 IID(dirichlet)的 fedavg/agent。这批此前已存档在 `docs/results/branch_result_collection_20260626/`,此处再汇总一份方便统一查阅。
- `agent/` —— 2026-06-30 那批(新):clean、het02 各方法,以及 het04 的 **agentmu(自适应 μ)** 3 个 seed、**muonly**、非 IID 的 **dir_agentmu**。这批是首次进仓库。

核心指标索引见 [`INDEX.md`](./INDEX.md)。所有结论以 **macro_auroc / macro_ap** 为准(阈值无关、多标签稳健);best_micro_f1 受阈值选择影响大,只作参考。

## 关键结论(多 seed 均值)

| 条件 | FedAvg | Robust-FedProx | Agent(门控) | AgentMu(门控+自适应μ) |
|---|---|---|---|---|
| clean(无噪声) | 0.825 | — | 0.827 | — |
| het02(异质噪声20%) | 0.721 | 0.730 | **0.781** | — |
| het04 IID(异质噪声40%) | 0.738 | 0.669 | **0.804** | 0.805 |
| het04 非IID(dirichlet) | 0.588 | — | 0.533 ⚠️ | 0.604 |

- **优势**:在 **IID + 异质标签噪声** 下,agent 门控聚合显著优于 FedAvg(+0.066)和静态鲁棒方法 Robust-FedProx(+0.135);clean 下不掉点(无副作用)。
- **自适应 μ**:在 IID 下与单纯门控基本持平(冗余);在非 IID 下略有回收(0.533→0.604),但仍未追平 FedAvg。
- **已知短板**:**非 IID(dirichlet)下门控会塌陷**——因为客户端探针分数差异被数据分布差异污染,误把"分布不同"当成"噪声"压低权重。→ 这正是补充实验里 **weight-floor(权重下限)** 和 **CCR/RHFL 基线** 要解决/对照的问题。

> 这两批结果现都在分支 `feat/agent-orchestration`。补充实验在新分支 `feat/agent-supplement`。
