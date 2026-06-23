#!/usr/bin/env bash
# =============================================================================
# fl-agent ONE-COMMAND runner
# =============================================================================
# 把这个仓库 clone 到任意机器后，运行:
#
#     ./run.sh
#
# 它会自动:
#   1. 安装依赖
#   2. (可选) 读取 HF token，做环境预检
#   3. 下载并校验完整 RFMiD 数据集 (已存在文件自动跳过)
#   4. 生成联邦划分 (IID / Dirichlet / domain-hash, seed 0/1/2)
#   5. 下载 RETFound 权重并运行论文矩阵 (有 token+GPU 时)，否则跑 fallback
#   6. 汇总所有结果到 docs/results/ 和 runs/
#
# 数据集和模型权重 **不放进 Git**，而是运行时从官方源下载 —— 这是合法且干净的做法。
#
# RETFound 权重是 gated 仓库，需要你先做两件事 (只能你本人做):
#   - 在 https://huggingface.co/YukunZhou/RETFound_mae_natureCFP 同意条款
#   - export HF_TOKEN=hf_xxx   (从 https://huggingface.co/settings/tokens 创建)
#
# 可用环境变量:
#   MODE=auto|retfound|fallback   (默认 auto: 有 token 跑 RETFound, 否则 fallback)
#   HF_TOKEN=hf_xxx               (RETFound gated 访问所需)
#   RFMID_DATA_DIR=...            (默认 data/raw/rfmid_full)
#   SEEDS="0 1 2"                 (生成划分的随机种子)
#   SKIP_INSTALL=1               (跳过 pip install)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-auto}"
DATA_DIR="${RFMID_DATA_DIR:-data/raw/rfmid_full}"
SEEDS="${SEEDS:-0 1 2}"

# ---- 0. 读取 .env (如果存在) -------------------------------------------------
if [[ -f .env ]]; then
  echo "==> Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ---- 1. 安装依赖 -------------------------------------------------------------
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  echo "==> [1/6] Installing dependencies (data, torch, paper)"
  python3 -m pip install -e ".[data,torch,paper]"
else
  echo "==> [1/6] SKIP_INSTALL=1, skipping pip install"
fi

# ---- 2. 决定运行模式 ---------------------------------------------------------
HAS_TOKEN=0
if [[ -n "${HF_TOKEN:-}${HUGGINGFACE_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  HAS_TOKEN=1
fi

RUN_RETFOUND=0
case "${MODE}" in
  retfound) RUN_RETFOUND=1 ;;
  fallback) RUN_RETFOUND=0 ;;
  auto)     RUN_RETFOUND="${HAS_TOKEN}" ;;
  *) echo "Unknown MODE=${MODE} (use auto|retfound|fallback)"; exit 2 ;;
esac

echo "==> [2/6] Mode=${MODE} -> $([[ "${RUN_RETFOUND}" == "1" ]] && echo RETFound || echo fallback-MLP)"
if [[ "${RUN_RETFOUND}" == "1" ]]; then
  echo "    Running preflight check (GPU + token + gated access)..."
  if ! python3 -m fed_agent.tools.check_env; then
    echo ""
    echo "!! Preflight FAILED for RETFound mode."
    echo "!! Fix the FAIL items above, or run fallback:  MODE=fallback ./run.sh"
    exit 1
  fi
fi

# ---- 3. 下载 + 校验数据 ------------------------------------------------------
echo "==> [3/6] Downloading + validating full RFMiD into ${DATA_DIR}"
python3 -m fed_agent.tools.export_hf_rfmid_subset \
  --split all --out_dir "${DATA_DIR}" --max_samples 0 --validate

# ---- 4. 生成划分 ------------------------------------------------------------
echo "==> [4/6] Building splits for seeds: ${SEEDS}"
for seed in ${SEEDS}; do
  python3 -m fed_agent.tools.build_splits \
    --labels_csv "${DATA_DIR}/train/labels.csv" \
    --out_dir configs/splits/generated \
    --seed "${seed}" \
    --n_clients 4 \
    --alphas 0.1 0.5 1.0
done

# ---- 5. 运行矩阵 ------------------------------------------------------------
# PAPER_MATRIX_YAML overrides the auto-selected matrix (e.g. the medium config).
if [[ -n "${PAPER_MATRIX_YAML:-}" ]]; then
  MATRIX_YAML="${PAPER_MATRIX_YAML}"
  STEM="$(basename "${MATRIX_YAML}" .yaml)"
  OUT_DIR="${PAPER_OUT_DIR:-runs/paper_matrix/${STEM}}"
  OUT_MD="${PAPER_OUT_MD:-docs/results/${STEM}.md}"
elif [[ "${RUN_RETFOUND}" == "1" ]]; then
  MATRIX_YAML="configs/paper_matrix.yaml"
  OUT_DIR="runs/paper_matrix/retfound"
  OUT_MD="docs/results/paper_matrix_retfound.md"
else
  MATRIX_YAML="configs/paper_matrix_full46_mlp.yaml"
  OUT_DIR="runs/paper_matrix/full46_mlp"
  OUT_MD="docs/results/paper_matrix_full46_mlp.md"
fi

echo "==> [5/6] Running matrix: ${MATRIX_YAML}"
python3 -m fed_agent.tools.run_paper_matrix \
  --matrix_yaml "${MATRIX_YAML}" \
  --out_dir "${OUT_DIR}"

# ---- 6. 汇总 ---------------------------------------------------------------
echo "==> [6/6] Summarizing -> ${OUT_MD}"
python3 -m fed_agent.tools.summarize_paper_matrix \
  "${OUT_DIR}/summary.json" \
  --out_md "${OUT_MD}" \
  --out_csv "${OUT_DIR}/summary.csv"

echo ""
echo "============================================================"
echo "DONE. Results:"
echo "  - ${OUT_MD}"
echo "  - ${OUT_DIR}/summary.{json,csv}"
echo "============================================================"
if [[ "${RUN_RETFOUND}" == "1" ]]; then
  echo "Sanity: every row should show RETFound=True."
else
  echo "NOTE: fallback (MLP) results. RETFound=False rows are NOT paper results."
  echo "      To get real RETFound results: set HF_TOKEN and re-run ./run.sh"
fi
