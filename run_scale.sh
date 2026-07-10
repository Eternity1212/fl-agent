#!/usr/bin/env bash
# =============================================================================
# 一键运行 "scale / 鲁棒基线" 矩阵 (GPU)
#   - median / trimmed-mean 拜占庭鲁棒聚合基线 (K=4, het04 + het04_dir)
#   - K=8 scalability (核心 5 方法, seed0 判决性)
# =============================================================================
# 一条龙: 装依赖 -> 预检 -> 下载RFMiD -> 生成 K4 & K8 split -> 跑矩阵 -> 汇总
#
# 用法 (在已 clone 仓库根目录, 已 export HF_TOKEN 或 RETFOUND_CKPT_PATH):
#   git fetch origin
#   git checkout feat/agent-supplement          # 若还没在此分支
#   export HF_TOKEN=hf_xxx                       # 或 export RETFOUND_CKPT_PATH=/path/RETFound_mae_natureCFP.pth
#   ./run_scale.sh                               # 全套
#
# 子命令:
#   ./run_scale.sh smoke   # 本地CPU快速验证 median/trimmed 分支 (不需GPU/权重)
#   ./run_scale.sh data    # 只做数据下载 + K4/K8 split 生成
#   ./run_scale.sh run     # 跳过数据准备, 只跑矩阵 + 汇总 (数据已就绪时)
#
# 环境变量: RFMID_DATA_DIR / OUT / SKIP_INSTALL=1
# 断点续跑: 每个 run 单独存盘, 中断重跑自动跳过已完成的 run。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"

MODE="${1:-all}"
DATA_DIR="${RFMID_DATA_DIR:-data/raw/rfmid_full}"
OUT="${OUT:-runs/paper_matrix/agent_scale}"
MATRIX="configs/paper_matrix_agent_scale.yaml"
SMOKE_OUT="runs/paper_matrix/agent_scale_smoke"

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
  echo " 生成 K=4 split (seeds 0 1 2) —— median/trimmed 用"
  echo "==================================================================="
  for seed in 0 1 2; do
    python3 -m fed_agent.tools.build_splits \
      --labels_csv "${DATA_DIR}/train/labels.csv" \
      --out_dir configs/splits/generated \
      --seed "${seed}" --n_clients 4 --alphas 0.1 0.5 1.0
  done

  echo "==================================================================="
  echo " 生成 K=8 split (seeds 0 1 2) —— scalability + 误差棒用"
  echo "==================================================================="
  for seed in 0 1 2; do
    python3 -m fed_agent.tools.build_splits \
      --labels_csv "${DATA_DIR}/train/labels.csv" \
      --out_dir configs/splits/generated \
      --seed "${seed}" --n_clients 8 --alphas 0.1 0.5 1.0
  done
}

run_smoke() {
  echo "==================================================================="
  echo " 本地 smoke (CPU): 验证 median / trimmed 聚合分支可跑通"
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
  echo " 跑 scale 矩阵 (26 run) -> ${OUT}"
  echo "==================================================================="
  python3 -m fed_agent.tools.run_paper_matrix \
    --matrix_yaml "${MATRIX}" \
    --out_dir "${OUT}"
  echo "==================================================================="
  echo " 汇总 (median/trimmed 基线 + K=8 五方法) + 导出 CSV"
  echo "==================================================================="
  python3 -m fed_agent.tools.summarize_agent "${OUT}/summary.json" --csv "${OUT}/scale_full.csv"
  echo "--- 生成图表 -> docs/figures/scale ---"
  python3 -m fed_agent.tools.make_agent_figures \
    --results_dir "${OUT}" --out_dir docs/figures/scale || \
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
    echo "DONE. 结果在: ${OUT}/  (summary.json + scale_full.csv + 各 run json)"
    echo "============================================================"
    ;;
  *) echo "未知模式: ${MODE} (用 all | data | run | smoke)"; exit 1 ;;
esac
