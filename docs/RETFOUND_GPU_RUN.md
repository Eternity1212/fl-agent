# 在 GPU 机器上跑真正的 RETFound + LoRA 论文实验

本文件是给「迁移到带 GPU 的机器」准备的操作手册。按顺序做完，就能跑出
`RETFound=True` 的论文级主结果。**全程不需要把账号密码给任何人**，只需要一个
Hugging Face token。

---

## 0. 为什么需要这一步

RETFound 的官方权重 `YukunZhou/RETFound_mae_natureCFP` 在 Hugging Face 上是
**gated（受限）** 仓库：

- 必须先在网页上「同意条款」拿到访问权限（**这一步只能你本人点，我无法代办**）。
- 之后用 **HF token** 才能下载权重。

在没有权限的机器上，代码会把结果标记成 `RETFound=False`（fallback，只证明流程能跑，
不能写进论文）。本文件帮你拿到权限并跑出 `RETFound=True` 的结果。

---

## 1. 申请 RETFound 访问权限（你本人操作，约 2 分钟）

1. 登录 Hugging Face 账号。
2. 打开 https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
3. 页面会显示「You need to agree to share your contact information」，
   填写表单并点击同意。
4. 同意后，页面应能看到文件列表（说明权限已开通）。

> 该权重协议是 CC BY-NC 4.0（非商用），论文研究用途没问题。

## 2. 创建 HF token（你本人操作，约 1 分钟）

1. 打开 https://huggingface.co/settings/tokens
2. 新建一个 **Read** 权限的 token，复制下来（形如 `hf_xxx`）。

## 3. 在 GPU 机器上设置 token

把仓库拷到 GPU 机器后，二选一：

方式 A（推荐，临时环境变量）：

```bash
export HF_TOKEN=hf_你的token
```

方式 B（用 .env 文件）：

```bash
cp .env.example .env
# 编辑 .env，把 HF_TOKEN 填进去
set -a; source .env; set +a
```

方式 C（官方 CLI 登录，等价）：

```bash
huggingface-cli login --token hf_你的token
```

---

## 4. 环境依赖

GPU 机器上建议用与 CUDA 匹配的 PyTorch。先装 PyTorch（按你的 CUDA 版本，
参考 https://pytorch.org/get-started/locally/ ），例如 CUDA 12.1：

```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

再装本项目其余依赖：

```bash
python3 -m pip install -e ".[data,paper]"
```

> 一键脚本里也会执行 `pip install -e ".[data,torch,paper]"`，但 GPU 版 torch
> 最好按上面命令先单独装好，避免装成 CPU 版。

---

## 5. 先做预检（强烈建议）

在开始长训练前，先确认 GPU、token、gated 访问都 OK：

```bash
python3 -m fed_agent.tools.check_env
```

期望输出里这几项都是 `PASS`：

```
[PASS] GPU: CUDA GPUs: NVIDIA A100 80GB ...
[PASS] timm: timm 1.0.x
[PASS] HF token: HF token present (***xxxx)
[PASS] RETFound access: RETFound checkpoint downloadable: /.../RETFound_mae_natureCFP.pth
RESULT: READY. You can run scripts/run_paper_gpu.sh
```

如果 `RETFound access` 是 `FAIL`：回到第 1 步确认网页已同意条款，并确认 token 正确。

---

## 6. 一键运行

```bash
export HF_TOKEN=hf_你的token   # 若已 source .env 可跳过
./scripts/run_paper_gpu.sh
```

脚本会依次：

1. 安装依赖
2. 预检环境
3. 下载并校验完整 RFMiD（train=1920 / validation=640 / test=640，已存在文件会跳过）
4. 生成 seed 0/1/2 的 IID / Dirichlet / domain-hash 划分
5. 运行 `configs/paper_matrix.yaml`（`require_retfound: true`，没权限会直接报错）
6. 汇总结果

---

## 6.5 先跑中等规模验证方向（推荐先做这一步）

直接跑完整多 seed 矩阵很贵。建议先用中等规模配置（seed 0、多 epoch/round、
注入标签噪声做鲁棒性对比）确认"抗噪鲁棒性 + LoRA 通信效率"这个方向真的有收益：

```bash
export HF_TOKEN=hf_你的token
./scripts/run_paper_medium.sh    # 单 A100 约 1-1.5 小时
```

它会跑：

- 干净 non-IID（Dirichlet α=0.5）下：Centralized / Local-only / FedAvg / FedProx / Robust-FedProx
- 注入训练标签噪声 p=0.2 下：FedAvg / FedProx / Robust-FedProx

看 `docs/results/paper_matrix_retfound_medium.md`：

- **鲁棒性**：从 noise=0 到 noise=0.2，Robust-FedProx 的 best macro-F1 下降幅度
  应明显小于 FedAvg/FedProx，才说明方向成立。
- **通信效率**：联邦行的 `bytes`（只传 LoRA adapter）应远小于全模型，且性能接近
  Centralized 上限。

如果这一步看到正向信号，再扩成完整多 seed 主矩阵；否则需要先迭代方法。

## 7. 产物在哪

- `docs/results/paper_matrix_retfound.md` — 可提交的结果表（含 method mean ± std）
- `runs/paper_matrix/retfound/summary.{json,csv}` — 原始结果（被 gitignore）
- `runs/paper_matrix/retfound/<run_name>.json` — 每条 run 的明细（支持断点续跑，
  重跑脚本会自动跳过已完成的 run）

**验收标准**：表里每一行的 `RETFound` 列都应是 `True`。只要有一行是 `False`，
说明那条用的是 fallback，不能作为论文结果，需要回头检查 token / gated 权限。

---

## 8. 显存不够 / OOM 怎么办

`configs/paper_matrix.yaml` 里可调（ViT-Large + LoRA 通常 16GB+ 够用 batch 8）：

- `batch_size`: 显存小就调到 4 或 2。
- `lora_rank`: 4 / 8 / 16，越小越省显存。
- `image_size`: 保持 `[224, 224]`（RETFound 预训练输入）。

改完直接重跑脚本即可（已完成的 run 会跳过）。

---

## 9. 扩展到完整论文主矩阵

`configs/paper_matrix.yaml` 目前是 seed=0 的强 baseline 小矩阵，用于先在 GPU 上
验证 RETFound 能跑通。确认正向后，可把它扩成论文主矩阵（参考
`docs/EXPERIMENTS.md` 第 4 节）：

- seeds = {0, 1, 2}
- splits = {IID, Dirichlet α=0.5, Dirichlet α=0.1, domain-hash}
- methods = {Centralized, Local-only, FedAvg, FedProx, Robust-FedProx}
- 每条 run 用 `split_json:` 字段指定对应划分文件
  （`configs/splits/generated/labels__*_K4_S<seed>.json`）。

`configs/paper_matrix_full46_mlp.yaml` 是同结构的多 seed 示例，可直接照抄它的
写法把 backbone 换成 `retfound_mae_vit_large`、补全 splits 与 seeds。

---

## 10. 我（助手）能不能帮你在本机直接验证？

可以，但有限制：

- 如果你愿意贴一个 **Read 权限的临时 token**，我可以在当前这台机器上跑
  `check_env` 和一次「构建 RETFound 模型 + 单 batch」的 sanity，确认 gated 权限
  和加载逻辑没问题。
- 但当前这台机器**没有 GPU**，ViT-Large 全量训练会非常慢，所以真正的论文主结果
  仍建议在 GPU 机器上用本手册跑。
- token 属于敏感信息，用完请到 https://huggingface.co/settings/tokens 撤销。
