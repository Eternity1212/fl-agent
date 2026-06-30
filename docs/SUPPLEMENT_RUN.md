# 补充实验运行指南 (feat/agent-supplement)

本分支在主结果之上补齐论文所缺的对照和短板修复。**你只需要在 GPU 机器上拉这个分支、跑一条命令。**

## 一、为什么要补这些实验(对应已知缺口)

| 缺口 | 现状 | 本分支补的内容 |
|---|---|---|
| 部分条件只有 1 个 seed | clean / het02 只有 seed0,无法报 mean±std | 补 s1/s2,凑齐 3 seed |
| 缺学界公认的鲁棒 FL 基线 | 只比了 FedAvg / Robust-FedProx | 新增 **RHFL/CCR**(客户端置信度 softmax 重加权,Fang & Ye CVPR'22) |
| 非 IID 下 agent 门控塌陷 | het04_dir 上 agent(0.53)< FedAvg(0.59) | 新增 **weight_floor(权重下限)**,防止门控把客户端压到 0;3 seed 验证 |
| 缺中等异质度对照 | 只有极端 a=0.1 | 新增 a=0.5 一组 |
| tau / muonly 等消融未跑完 | tau002/tau005 没跑;muonly、dir_agentmu 只有 s0 | 补 tau002_s0/tau005_s0;muonly、dir_agentmu 补到 3 seed |

共 **37 个 run**,与主矩阵同设(RETFound+LoRA、40 轮、探针用 validation、final 在 test)。

## 跑完后的 seed 完整性(关键)

**所有主对比都会达到 3 seed(可报 mean±std);只有超参 sweep / 短板探针类按惯例保留 1 seed。**

| 条件 | 跑完后 3-seed 的方法 | 仅 1-seed(ablation/sweep,惯例如此) |
|---|---|---|
| clean | fedavg, agent | — |
| het02 | fedavg, robust, **ccr**, agent | — |
| het04 (IID) | fedavg, robust, **ccr**, agent, agentmu, muonly | tau002, tau005(tau 软硬 sweep) |
| het04_dir (a=0.1) | fedavg, agent, agentmu, **floor**, **agentmu+floor** | ccr, floor03(floor 强度 sweep) |
| het04_dir05 (a=0.5) | — | fedavg, agent, floor(中等异质探针) |

> 这意味着论文主表(FedAvg / Robust-FedProx / RHFL-CCR / Agent / Agent+floor,跨 clean/het02/het04/het04_dir)**全部 3 seed**。tau / floor 强度 / a=0.5 这些是"调参曲线"和"补充探针",单 seed 即可,论文里不需要 std。

## 二、一键运行(在 GPU 机器)

```bash
# 1. 进入已下载好数据/权重的仓库目录
cd fl-agent

# 2. 拉取并切到补充分支
git fetch origin
git checkout feat/agent-supplement

# 3. 设置 RETFound 权重访问(二选一)
export HF_TOKEN=hf_xxx
# 或: export RETFOUND_CKPT_PATH=/path/to/RETFound_mae_natureCFP.pth

# 4. 一键跑:装依赖 -> 下载/校验RFMiD -> 生成split -> 跑31个run -> 汇总+图表
./run_supplement.sh
```

> 如果数据和 split 之前已经准备好,可直接 `./run_supplement.sh run` 跳过下载步骤。
> 断点续跑:每个 run 单独存盘,中断后重跑自动跳过已完成的。

## 三、跑之前先本地 smoke(可选,~1 分钟,不需 GPU)

确认新增的 ccr / weight_floor 代码分支没问题:

```bash
./run_supplement.sh smoke
```

## 四、跑完看结果

```bash
# 文字 verdict(每个条件下各方法 macro_auroc mean±std + 是否 WINS)
python3 -m fed_agent.tools.summarize_agent runs/paper_matrix/agent_supp/summary.json

# 图表(权重轨迹 + AUROC 柱状) 在 docs/figures/supp/
```

## 五、本分支代码改动一览

- `src/fed_agent/agent/orchestrator.py`
  - 新增 `decide_weights_ccr(...)`:RHFL/CCR 基线(softmax 置信度加权)
  - `decide_weights(...)` 新增 `weight_floor` 参数:`gate_eff = floor + (1-floor)*gate`,防非 IID 塌陷
- `src/fed_agent/train/paper_runner.py`:新增配置 `agent_ccr_temp`、`agent_weight_floor`,并在 `agent_aggregation: ccr` 时走 CCR 分支
- `configs/paper_matrix_agent_supp.yaml`:31 个补充 run
- `run_supplement.sh`:一键脚本
- `tests/test_agent_orchestrator.py`:新增 ccr / weight_floor 单测(已通过)
