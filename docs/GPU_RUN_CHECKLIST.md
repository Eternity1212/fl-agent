# GPU 服务器跑论文实验 · Checklist

照着从上到下打勾即可。两种权重获取方式二选一（A 推荐：上传本地权重，无需 token）。

---

## ☐ 0. 前置确认

- [ ] 服务器有 NVIDIA GPU（建议 A100 / ≥24GB 显存）
- [ ] 能联网（要从公开 HF 镜像下载 RFMiD 数据集）
- [ ] 已安装 Python 3.10+、git

---

## ☐ 1. 拉取代码

```bash
git clone git@github.com:Eternity1212/fl-agent.git
cd fl-agent
```

> 用 HTTPS 也行：`git clone https://github.com/Eternity1212/fl-agent.git`

---

## ☐ 2. 安装 CUDA 版 PyTorch（必须先装，避免装成 CPU 版）

```bash
# 按服务器 CUDA 版本选 index-url，这里以 CUDA 12.1 为例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> 其余依赖 `run.sh` 会自动装；不想让它重装 torch 可在第 5 步加 `SKIP_INSTALL=1`
> 并先手动 `pip install -e ".[data,paper]"`。

---

## ☐ 3. 准备 RETFound 权重（A / B 二选一）

### 方式 A（推荐）：上传本地已下载的权重，服务器无需 HF token

本机权重路径：
```
/Users/bytedance/.cache/huggingface/hub/models--YukunZhou--RETFound_mae_natureCFP/snapshots/556830f78214f0e8da35af965292ac5a3180ac47/RETFound_mae_natureCFP.pth
```

上传（约 3.95 GB）：
```bash
scp "/Users/bytedance/.cache/huggingface/hub/models--YukunZhou--RETFound_mae_natureCFP/snapshots/556830f78214f0e8da35af965292ac5a3180ac47/RETFound_mae_natureCFP.pth" \
    user@gpu-server:/data/weights/RETFound_mae_natureCFP.pth
```

在服务器上指向它：
```bash
export RETFOUND_CKPT_PATH=/data/weights/RETFound_mae_natureCFP.pth
```

- [ ] 权重已上传
- [ ] `RETFOUND_CKPT_PATH` 已 export

### 方式 B：用 HF token 在线下载（需已在网页同意 gated 条款）

```bash
export HF_TOKEN=hf_你的token
```

- [ ] 已在 https://huggingface.co/YukunZhou/RETFound_mae_natureCFP 同意条款
- [ ] `HF_TOKEN` 已 export

---

## ☐ 4. 预检环境（强烈建议）

```bash
python3 -m fed_agent.tools.check_env
```

期望关键项为 PASS：`GPU` / `timm` / `RETFound access`。
- [ ] 输出 `RESULT: READY`

---

## ☐ 5. 一键跑全套（对比 + 消融 + 多 seed），挂后台

```bash
nohup ./run.sh > run.log 2>&1 &
tail -f run.log     # 看进度，Ctrl-C 只退出查看，不影响后台
```

`run.sh` 自动：装依赖 → 预检 → 下载+校验完整 RFMiD → 生成划分 →
跑 `configs/paper_matrix_gpu_full.yaml`（29 个 run）→ 汇总结果。

- [ ] 后台已启动（记下 PID）
- [ ] `run.log` 在持续输出

> 预计单 A100 约 6–10 小时。中断后重跑 `./run.sh` 会自动跳过已完成的 run。

---

## ☐ 6. （可选）先跑中等规模验证方向（~1 小时）

不确定要不要全套时，先用中等配置快速看信号：
```bash
PAPER_MATRIX_YAML=configs/paper_matrix_retfound_medium.yaml ./run.sh
```

---

## ☐ 7. 取结果

- [ ] 报告：`docs/results/paper_matrix_retfound_full.md`（含 method mean ± std）
- [ ] 原始：`runs/paper_matrix/retfound_full/summary.{json,csv}`
- [ ] 明细：`runs/paper_matrix/retfound_full/<run_name>.json`

**验收**：表里每行 `RETFound` 列都应是 `True`。任一行是 `False` → 那条走了
fallback，不能进论文，回到第 3 步检查权重 / token。

---

## 矩阵内容速览（29 个 run）

| 区块 | 内容 |
|------|------|
| 对比 | centralized / local_only / fedavg / fedprox / robust_fedprox × seed 0,1,2 |
| 消融 B1 噪声 | fedavg vs robust @ noise {0.2, 0.4} |
| 消融 B2 非IID | fedavg / fedprox / robust @ Dirichlet {0.1, 0.5} |
| 消融 B3 LoRA rank | robust @ rank {4, 16} |
| 消融 B4 FedProx μ | fedprox @ μ {0.01, 0.1} |

---

## 显存不够 / OOM

改 `configs/paper_matrix_gpu_full.yaml` 的 `defaults`：
- `batch_size`: 16 → 8 → 4
- `lora_rank`: 8 → 4
- `image_size`: 保持 `[224, 224]`（RETFound 预训练输入，别改）

改完直接重跑 `./run.sh`（已完成的 run 自动跳过）。
