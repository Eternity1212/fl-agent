# Agent 实验交接记录

更新时间：2026-06-26
分支：`results/retfound-optimized-summary`
仓库根目录：`/opt/tiger/gushiyu123/fl/fl-agent`

## 1. 这次新增了哪些代码/配置/分析文件

相对当前 git HEAD，可见的 agent 相关新增/修改主要包括：

### 修改过的已跟踪文件
- `src/fed_agent/tools/run_paper_matrix.py`
- `src/fed_agent/train/paper_runner.py`

### 新增的 agent 相关文件/目录
- `configs/paper_matrix_agent.yaml`
- `configs/paper_matrix_agent_stage1.yaml`
- `configs/paper_matrix_agent_smoke.yaml`
- `configs/paper_matrix_agent_smoke3.yaml`
- `docs/AGENT_EXPERIMENTS.md`
- `docs/PAPER_OUTLINE.md`
- `inspect_agent_run.py`
- `run_agent.sh`
- `scripts/agent_smoke.py`
- `src/fed_agent/agent/__init__.py`
- `src/fed_agent/agent/orchestrator.py`
- `src/fed_agent/tools/summarize_agent.py`
- `tests/test_agent_orchestrator.py`

### 当前相关结果文件
#### stage1 已完成结果
目录：`runs/paper_matrix/agent_stage1/`
- `het04_fedavg_s0.json`
- `het04_fedavg_s1.json`
- `het04_fedavg_s2.json`
- `het04_robust_s0.json`
- `het04_robust_s1.json`
- `het04_robust_s2.json`
- `het04_agent_s0.json`
- `het04_agent_s1.json`
- `het04_agent_s2.json`
- `het04_dir_fedavg_s0.json`
- `het04_dir_agent_s0.json`
- `summary.json`

#### full matrix 新增已落盘结果
目录：`runs/paper_matrix/agent/`
- `clean_fedavg_s0.json`
- `clean_agent_s0.json`
- `het02_fedavg_s0.json`
- `het02_robust_s0.json`
- `het02_agent_s0.json`

## 2. 目前实验跑到哪里

### full agent matrix 定义
配置：`configs/paper_matrix_agent.yaml`
总 run 数：23

### 已完成
共 16/23：
- `het04_fedavg_s0`
- `het04_fedavg_s1`
- `het04_fedavg_s2`
- `het04_robust_s0`
- `het04_robust_s1`
- `het04_robust_s2`
- `het04_agent_s0`
- `het04_agent_s1`
- `het04_agent_s2`
- `het04_dir_fedavg_s0`
- `het04_dir_agent_s0`
- `clean_fedavg_s0`
- `clean_agent_s0`
- `het02_fedavg_s0`
- `het02_robust_s0`
- `het02_agent_s0`

### 仍待完成
共 7 个：
- `het04_muonly_s0`
- `het04_agentmu_s0`
- `het04_agentmu_s1`
- `het04_agentmu_s2`
- `het04_dir_agentmu_s0`
- `het04_agent_tau002_s0`
- `het04_agent_tau005_s0`

## 3. 已跑出来的关键结果

### 3.1 stage1 核心结论：IID heterogeneous noise 下 agent 明显有效

条件：`het04`（IID + 客户端 2/3 异质噪声 0.4）

#### FedAvg macro-AUROC
- s0 = 0.7374030006713008
- s1 = 0.7376699869658251
- s2 = 0.7389012582339941
- 均值约 = **0.7380**

#### Robust macro-AUROC
- s0 = 0.6668525498002645
- s1 = 0.6761687196960625
- s2 = 0.6637292364135062
- 均值约 = **0.6689**

#### Agent macro-AUROC
- s0 = 0.7907738471449903
- s1 = 0.8102057656520006
- s2 = 0.8120244705917075
- 均值约 = **0.8043**

#### 结论
- Agent 相对 FedAvg 的提升约 = **+0.0663 macro-AUROC**
- 这不只是过 `0.76`，而是稳定拉到 `0.79 ~ 0.81`
- Robust-FedProx 在这个设定下明显差于 FedAvg

### 3.2 机制证据
使用 `inspect_agent_run.py` 查看可知：
- 在 `het04_agent_s0/s1/s2` 中，clean client 0/1 平均权重约 `0.5 / 0.5`
- noisy client 2/3 平均权重约 `0 / 0`
- probe score 对 clean/noisy 有明显分层

说明：
- agent 不是偶然涨点
- 它确实识别并压低了脏客户端更新

### 3.3 non-IID 下当前纯 agent 失败
条件：`het04_dir`
- `het04_dir_fedavg_s0` macro-AUROC = **0.5876**
- `het04_dir_agent_s0` macro-AUROC = **0.5330**

结论：
- 当前“纯自适应加权”在 non-IID + hetero noise 下会输给 FedAvg
- 后续最关键待验证 run：`het04_dir_agentmu_s0`

### 3.4 clean 控制组已有两条结果
#### `clean_fedavg_s0`
- `macro_auroc = 0.8249479088384648`
- `macro_ap = 0.2796403644086115`
- `micro_f1 = 0.648363252375924`
- `best_micro_f1 = 0.6598763186613313`
- `best_macro_f1_present = 0.25392789353886475`

#### `clean_agent_s0`
- `macro_auroc = 0.8264744667280821`
- `macro_ap = 0.29054721780140047`
- `best_micro_f1 = 0.693567718737515`
- `best_macro_f1_present = 0.27067920554470015`

结论：
- clean 下 agent **没有掉点**，反而与 clean FedAvg 非常接近并略高
- 这支持“clean 时退化到强基线附近”的主张

### 3.5 温和异质噪声 het02 已完成三方法对比
#### `het02_fedavg_s0`
- `macro_auroc = 0.7210696941982812`
- `macro_ap = 0.24261623017201195`
- `best_macro_f1_present = 0.15490287893627674`
- `best_micro_f1 = 0.3035475234270415`

#### `het02_robust_s0`
- `macro_auroc = 0.73020447220112`
- `macro_ap = 0.19144954447054446`
- `best_macro_f1_present = 0.15566148030610946`
- `best_micro_f1 = 0.5955940204563336`

#### `het02_agent_s0`
- `macro_auroc = 0.7812136986942531`
- `macro_ap = 0.2663042046792261`
- `best_macro_f1_present = 0.2384586353958998`
- `best_micro_f1 = 0.6589810338415768`

结论：
- 在温和异质噪声下，agent 仍然明显优于 fedavg
- 相对 fedavg 的 macro-AUROC 提升约 = **+0.0601**
- robust 在 macro-AUROC 上略高于 fedavg，但整体仍明显弱于 agent

## 4. 当前运行中的任务

目前已经用独立环境启动了剩余实验批次，命令逻辑相当于：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-agent/bin/python -m fed_agent.tools.run_paper_matrix \
  --matrix_yaml configs/paper_matrix_agent.yaml \
  --out_dir runs/paper_matrix/agent \
  --only clean_fedavg_s0 clean_agent_s0 het02_fedavg_s0 het02_robust_s0 het02_agent_s0 \
         het04_muonly_s0 het04_agentmu_s0 het04_agentmu_s1 het04_agentmu_s2 \
         het04_dir_agentmu_s0 het04_agent_tau002_s0 het04_agent_tau005_s0
```

如果实例释放前任务未跑完，需要在新实例上重新检查 `runs/paper_matrix/agent/` 已落盘哪些 json，再继续跑缺失项即可。`run_paper_matrix` 会跳过已完成 run。

## 5. 如何在新实例继续跑

### 5.1 先准备环境（重要）
这次最大的坑是共享 Python 环境已经损坏，不要再直接复用用户 site-packages。

推荐使用独立环境：

```bash
python3 -m pip install --user --index-url https://pypi.org/simple virtualenv
python3 -m virtualenv .venv-agent
```

然后安装依赖（这里使用 public PyPI，因为内部源会 403）：

```bash
.venv-agent/bin/python -m pip install --index-url https://pypi.org/simple \
  torch torchvision timm scikit-learn PyYAML tqdm numpy huggingface_hub
PIP_NO_BUILD_ISOLATION=1 .venv-agent/bin/python -m pip install --no-build-isolation -e .
```

### 5.2 验证环境

```bash
CUDA_VISIBLE_DEVICES=0 .venv-agent/bin/python -m fed_agent.tools.check_env
```

期望：
- `GPU: PASS`
- `timm: PASS`
- `RETFound access: PASS`
- 即使 `HF token` fail，只要本地 cache checkpoint 可访问，当前这批实验仍可继续跑

### 5.3 继续跑剩余实验
在新实例上，先看哪些文件已经存在：

```bash
find runs/paper_matrix/agent -maxdepth 1 -type f | sort
```

然后只跑缺失项，例如：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-agent/bin/python -m fed_agent.tools.run_paper_matrix \
  --matrix_yaml configs/paper_matrix_agent.yaml \
  --out_dir runs/paper_matrix/agent \
  --only <缺失的 run 名称们>
```

### 5.4 跑完后汇总

```bash
.venv-agent/bin/python -m fed_agent.tools.summarize_agent runs/paper_matrix/agent/summary.json
python3 inspect_agent_run.py runs/paper_matrix/agent_stage1/*.json runs/paper_matrix/agent/*.json
```

## 6. 这次遇到的主要问题 / 注意点

### 6.1 共享环境漂移
之前 stage1 能跑，但后续续跑失败，根因是共享用户环境里的版本漂移：
- `torch` 被升级到 2.8.0
- `torchvision` 停在 0.18.1+cu121
- `timm` 导入时触发 `torchvision::nms does not exist`

报错表现：
- `RuntimeError: operator torchvision::nms does not exist`
- 或 `RuntimeError: Install paper extras: python3 -m pip install -e '.[paper]'`

### 6.2 不能依赖内部 pip 源
内部源 `https://bytedpypi.byted.org/simple/` 多次 403，导致：
- `virtualenv`
- `setuptools`
- `wheel`
等安装失败

解决方式：
- 显式使用 `--index-url https://pypi.org/simple`

### 6.3 `venv` 失败
系统缺 `ensurepip`，导致：
- `python3 -m venv .venv-agent` 失败

解决方式：
- 先装 `virtualenv`，再用 `python3 -m virtualenv .venv-agent`

### 6.4 build isolation 失败
`pip install -e .` 默认会尝试 build isolation，从内部源拉 `setuptools>=68`，会失败。

解决方式：

```bash
PIP_NO_BUILD_ISOLATION=1 .venv-agent/bin/python -m pip install --no-build-isolation -e .
```

### 6.5 训练结果不会实时刷 stdout
`run_paper_matrix.py` 基本不打印训练进度；每个 run 完整结束后才会写 `<run>.json`。
所以：
- 目录暂时没文件，不代表没在跑
- 应结合进程与 GPU 利用率判断是否仍在训练

## 7. 数据集大小，是否适合上传 GitHub

当前 `data/` 大小约：
- `data/` = **7.5G**
- `data/raw/` = **7.5G**

结论：
- **不适合上传 GitHub 仓库**
- 普通 GitHub 仓库对大文件非常不友好，7.5G 原始数据更不应该进 git 历史
- 即便用 Git LFS，也不建议把这种原始医学图像数据放进仓库

建议：
- 保留数据下载/准备步骤
- 在新实例上按已有流程重新准备数据，或从对象存储/数据盘同步
- 代码、配置、split json、结果 json、handoff 文档更适合进压缩包或单独保存

### 为什么不建议直接上传数据到 GitHub
1. 体积太大（7.5G）
2. 拉取/推送慢
3. 容易污染仓库历史
4. 数据许可与隐私边界也更敏感

结论：
- **上传代码与结果元数据，保留数据在外部存储，更高效**

## 8. 推荐打包的内容
建议把以下内容打包带走：
- agent 相关代码与配置
- 当前结果 json
- 生成的 split json
- 本交接文档

不建议打包原始 `data/raw/` 进仓库归档；若必须备份数据，单独做数据归档，不要混进代码仓库发布物。
