#!/usr/bin/env bash
# =============================================================================
# 一键运行 "补充实验" 矩阵 (GPU) —— 补齐论文所缺对照 + 修复非IID短板
# =============================================================================
# 这个脚本做完整的一条龙: 装依赖 -> 预检 -> 下载RFMiD -> 生成split -> 跑31个补充run
#                          -> 汇总 verdict -> 生成图表
#
# 用法 (在已 clone 仓库根目录, 已 export HF_TOKEN 或 RETFOUND_CKPT_PATH):
#
#   git fetch origin
#   git checkout feat/agent-supplement
#   export HF_TOKEN=hf_xxx            # 或 export RETFOUND_CKPT_PATH=/path/RETFound_mae_natureCFP.pth
#   ./run_supplement.sh              # 全套: 数据+split+矩阵+汇总+图表
#
# 子命令:
#   ./run_supplement.sh smoke        # 本地CPU快速验证管线 (含 ccr/floor 分支), 不需GPU/权重
#   ./run_supplement.sh data         # 只做数据下载 + split 生成
#   ./run_supplement.sh run          # 跳过数据准备, 只跑矩阵 + 汇总 (数据已就绪时)
#
# 环境变量:
#   RFMID_DATA_DIR=...   默认 data/raw/rfmid_full
#   SEEDS="0 1 2"        生成 split 的种子
#   OUT=...              结果输出目录 (默认 runs/paper_matrix/agent_supp)
#   SKIP_INSTALL=1       跳过 pip install
#
# 断点续跑: 每个 run 单独存盘, 中断重跑自动跳过已完成的 run。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"

MODE="${1:-all}"
DATA_DIR="${RFMID_DATA_DIR:-data/raw/rfmid_full}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-runs/paper_matrix/agent_supp}"
MATRIX="configs/paper_matrix_agent_supp.yaml"
SMOKE_OUT="runs/paper_matrix/agent_supp_smoke"

# ---- .env (可选) ----
if [[ -f .env ]]; then set -a; source .env; set +a; fi

install_deps() {
  if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo "==> 安装依赖 (.[data,torch,paper])"
    python3 -m pip install -e ".[data,torch,paper]"
  else
    echo "==> SKIP_INSTALL=1, 跳过 pip install"
  fi
}

prep_data() {
  echo "==================================================================="
  echo " 下载 + 校验完整 RFMiD -> ${DATA_DIR}  (已存在文件自动跳过)"
  echo "==================================================================="
  python3 -m fed_agent.tools.export_hf_rfmid_subset \
    --split all --out_dir "${DATA_DIR}" --max_samples 0 --validate

  echo "==================================================================="
  echo " 生成联邦 split (IID / Dirichlet a=0.1,0.5,1.0 / domain-hash), seeds: ${SEEDS}"
  echo "==================================================================="
  for seed in ${SEEDS}; do
    python3 -m fed_agent.tools.build_splits \
      --labels_csv "${DATA_DIR}/train/labels.csv" \
      --out_dir configs/splits/generated \
      --seed "${seed}" \
      --n_clients 4 \
      --alphas 0.1 0.5 1.0
  done
}

run_smoke() {
  echo "==================================================================="
  echo " 本地 smoke (CPU): 验证 ccr / weight_floor 分支可跑通"
  echo "==================================================================="
  python3 -m fed_agent.tools.run_paper_matrix \
    --matrix_yaml configs/paper_matrix_agent_smoke3.yaml \
    --out_dir "${SMOKE_OUT}"
  python3 -m fed_agent.tools.summarize_agent "${SMOKE_OUT}/summary.json"
}

run_matrix() {
  echo "==================================================================="
  echo " GPU 预检 (RETFound 权重: HF_TOKEN 或 RETFOUND_CKPT_PATH)"
  echo "==================================================================="
  python3 -m fed_agent.tools.check_env || {
    echo "!! 预检未通过: 请先 export HF_TOKEN=... 或 export RETFOUND_CKPT_PATH=/path/RETFound_mae_natureCFP.pth"
    exit 1
  }
  echo "==================================================================="
  echo " 跑补充矩阵 (31 run) -> ${OUT}"
  echo "==================================================================="
  python3 -m fed_agent.tools.run_paper_matrix \
    --matrix_yaml "${MATRIX}" \
    --out_dir "${OUT}"
  echo "==================================================================="
  echo " 汇总 verdict (fedavg / robust / ccr / agent / floor ...)"
  echo "==================================================================="
  python3 -m fed_agent.tools.summarize_agent "${OUT}/summary.json"
  echo "--- 生成图表 -> docs/figures/supp ---"
  python3 -m fed_agent.tools.make_agent_figures \
    --results_dir "${OUT}" --out_dir docs/figures/supp || \
    echo "(图表生成跳过: 需 pip install -e '.[paper]')"
}

case "${MODE}" in
  smoke) run_smoke ;;
  data)  install_deps; prep_data ;;
  run)   run_matrix ;;
  all)
    install_deps
    prep_data
    run_matrix
    echo ""
    echo "============================================================"
    echo "DONE. 结果在: ${OUT}/  (summary.json + 各 run json)"
    echo "图表在: docs/figures/supp/"
    echo "============================================================"
    ;;
  *) echo "未知模式: ${MODE} (用 all | data | run | smoke)"; exit 1 ;;
esac
