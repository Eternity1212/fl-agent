# Agent (Adaptive Orchestration) 实验说明

## 1. 动机:为什么需要 agent

主矩阵 (`paper_matrix_gpu_full.yaml`) 的结论:
- clean 下 **FedAvg 最强**,FedProx / Robust-FedProx 不如它(近端正则叠加在已被低秩 LoRA 强约束的参数上 → 双重正则 → 欠拟合)。
- 噪声 / non-IID 消融里 Robust 也没翻盘。

**但那些噪声消融是"均匀噪声"(所有客户端被同样污染)**,在这种设定下没有"干净客户端"可参照,任何方法都无从选择,FedAvg 靠平均降方差自然占优。

真实联邦医疗场景是 **异质 (heterogeneous)**:不同医院标注质量不同。agent 的价值就在这里:
- 每轮用共享探针集给"全局+各客户端本地更新"打分;
- 用绕中位数的 sigmoid 门控压低异常低分(脏)客户端,同时保持正常客户端均衡(不塌缩到单一客户端);
- **clean 时退化成 FedAvg(不掉点),异质噪声 / 异质 non-IID 下应优于 FedAvg 与 Robust。**

## 2. 实现

- 策略: `src/fed_agent/agent/orchestrator.py` 的 `decide_weights`
  - 探针分 = 负验证 BCE(比 AUROC 更敏感于标签噪声损伤)
  - `w_i ∝ size_i · sigmoid((s_i − median_s)/tau)`;分数全相等时门控全 0.5 → 退化为 size 加权(FedAvg)
- 联邦循环: `src/fed_agent/train/paper_runner.py` 的 `_run_agent_federated`(method=`agent_fed`)
  - 支持 per-client 标签噪声注入(`agent_noisy_clients` + `agent_client_noise`),用于异质噪声实验
  - 支持 fedprox 近端项(`fedprox_mu>0`)+ `positive_dropout`,因此同一 runner 可表达 FedAvg / Robust / Agent 三种方法,保证在**同一份异质噪声**下公平对比

## 3. 配置矩阵 `configs/paper_matrix_agent.yaml`

| 组 | 内容 |
|---|---|
| A. CLEAN 控制 | agent vs fedavg,noise=0 → 验证 agent 不掉点 |
| B. 异质噪声@0.4 | fedavg / robust / agent × seed 0/1/2(决定性对比) |
| C. 异质噪声@0.2 | fedavg / robust / agent × seed 0 |
| D. 异质噪声 + 非IID | agent vs fedavg @ Dirichlet α=0.1, seed 0 |
| E. tau 消融 | agent @ tau {0.02, 0.05}, seed 0 |

公平性:三种方法都走 `agent_fed` runner、同一 `agent_noisy_clients=[2,3]`、同一 noise 水平;只改聚合方式 / 正则。

## 4. 运行 (GPU)

```bash
# clone 后, 已 export HF_TOKEN 或设 RETFOUND_CKPT_PATH
python3 -m fed_agent.tools.run_paper_matrix \
  --matrix_yaml configs/paper_matrix_agent.yaml \
  --out_dir runs/paper_matrix/agent
```

断点续跑:每个 run 单独存盘,中断重跑自动跳过已完成 run。

本地快速验证(CPU/TinyCNN/子集,非论文数字):

```bash
python3 -m fed_agent.tools.run_paper_matrix \
  --matrix_yaml configs/paper_matrix_agent_smoke.yaml \
  --out_dir runs/paper_matrix/agent_smoke
```

## 5. 判读标准(决定性)

- **agent > FedAvg(异质噪声下,跨 seed 稳定)** → agent 是真方法贡献,论文成立。
- **agent ≈ FedAvg** → 至少"自适应不掉点 + 干净退化为 FedAvg",配负结果仍可写。
- **agent 也输** → 回到 benchmark + 负结果论文。

诊断信号(每个 agent run 已记录):
- `agent_weight_history`:每轮各客户端权重 → 看是否压低脏客户端 2,3
- `agent_probe_history`:每轮探针分 → 看干净/脏是否分层

## 6. v0 caveat(已知,待修)

当前 agent 探针用 **validation 集**打分,final 也在 validation 上报告,agent 对 val 有"偷看"。
这是 v0 快速验证版。**若 agent 显示收益,需用 held-out probe(或在 test 上报告 final)重跑确认**,以排除选择性泄漏。
