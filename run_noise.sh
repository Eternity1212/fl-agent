#!/usr/bin/env bash
# =============================================================================
# 一键运行 "噪声鲁棒" 两组实验 (GPU) —— 冲 MICCAI 主会的补强
#   (A) FedNoRo baseline  : 两阶段(预热+客户端loss GMM)+ 脏客户端 GCE 鲁棒损失
#   (B) 非对称(class-conditional)标签噪声 : 第二种噪声模型, fedavg/agent/ccr/feda3i
# =============================================================================
# 一条龙: 装依赖 -> 预检 -> 下载RFMiD -> 生成 K4 split(seed0,1,2) -> 跑矩阵 -> 汇总
#
# 用法 (在已 clone 的仓库根目录; 已 export HF_TOKEN 或 RETFOUND_CKPT_PATH):
#   git fetch origin
#   git checkout feat/agent-supplement          # 若还没在此分支
#   export HF_TOKEN=hf_xxx                       # 或 export RETFOUND_CKPT_PATH=/path/RETFound_mae_natureCFP.pth
#   ./run_noise.sh                               # 全套 (data + fednoro + asym)
#
# 子命令:
#   ./run_noise.sh smoke     # 本地CPU快速验证 fednoro/asym 分支 (不需GPU/权重)
#   ./run_noise.sh data      # 只下载 RFMiD + 生成 K4 split(seeds 0,1,2)
#   ./run_noise.sh fednoro   # 只跑 FedNoRo 矩阵(9 run) + 汇总
#   ./run_noise.sh asym      # 只跑 非对称噪声矩阵(24 run) + 汇总
#   ./run_noise.sh run       # 跑 fednoro + asym (跳过数据准备)
#
# 环境变量: RFMID_DATA_DIR / SKIP_INSTALL=1
# 断点续跑: 每个 run 单独存盘, 中断重跑自动跳过已完成的 run。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"

MODE="${1:-all}"
DATA_DIR="${RFMID_DATA_DIR:-data/raw/rfmid_full}"
FEDNORO_OUT="runs/paper_matrix/agent_fednoro"
ASYM_OUT="runs/paper_matrix/agent_asym"
SMOKE_OUT="runs/paper_matrix/agent_noise_smoke"
FEDNORO_MATRIX="configs/paper_matrix_agent_fednoro.yaml"
ASYM_MATRIX="configs/paper_matrix_agent_asym.yaml"

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
  echo " 生成 K=4 split (seeds 0 1 2): IID + Dirichlet a=0.1"
  echo "==================================================================="
  for seed in 0 1 2; do
    python3 -m fed_agent.tools.build_splits \
      --labels_csv "${DATA_DIR}/train/labels.csv" \
      --out_dir configs/splits/generated \
      --seed "${seed}" --n_clients 4 --alphas 0.1 0.5 1.0
  done
}

preflight() {
  echo "==================================================================="
  echo " GPU 预检 (RETFound 权重: HF_TOKEN 或 RETFOUND_CKPT_PATH)"
  echo "==================================================================="
  python3 -m fed_agent.tools.check_env || {
    echo "!! 预检未通过: 请先 export HF_TOKEN=... 或 export RETFOUND_CKPT_PATH=/path/RETFound_mae_natureCFP.pth"
    exit 1
  }
}

run_fednoro() {
  preflight
  echo "==================================================================="
  echo " 跑 FedNoRo 矩阵 (9 run: het02/het04/het04_dir x 3 seed) -> ${FEDNORO_OUT}"
  echo "==================================================================="
  python3 -m fed_agent.tools.run_paper_matrix \
    --matrix_yaml "${FEDNORO_MATRIX}" --out_dir "${FEDNORO_OUT}"
  python3 -m fed_agent.tools.summarize_agent "${FEDNORO_OUT}/summary.json" \
    --csv "${FEDNORO_OUT}/fednoro_full.csv"
}

run_asym() {
  preflight
  echo "==================================================================="
  echo " 跑 非对称噪声矩阵 (24 run: het04 IID + het04_dir x 4方法 x 3 seed) -> ${ASYM_OUT}"
  echo "==================================================================="
  python3 -m fed_agent.tools.run_paper_matrix \
    --matrix_yaml "${ASYM_MATRIX}" --out_dir "${ASYM_OUT}"
  python3 -m fed_agent.tools.summarize_agent "${ASYM_OUT}/summary.json" \
    --csv "${ASYM_OUT}/asym_full.csv"
}

run_smoke() {
  echo "==================================================================="
  echo " 本地 smoke (CPU): 验证 fednoro / asym 聚合分支可跑通 (不需GPU/权重)"
  echo "==================================================================="
  python3 -m fed_agent.tools.run_paper_matrix \
    --matrix_yaml configs/paper_matrix_agent_smoke3.yaml \
    --out_dir "${SMOKE_OUT}" --only het04_fednoro_s0 het04asym_agent_s0
  python3 -m fed_agent.tools.summarize_agent "${SMOKE_OUT}/summary.json"
}

case "${MODE}" in
  smoke)   run_smoke ;;
  data)    install_deps; prep_data ;;
  fednoro) run_fednoro ;;
  asym)    run_asym ;;
  run)     run_fednoro; run_asym ;;
  all)
    install_deps
    prep_data
    run_fednoro
    run_asym
    echo ""
    echo "============================================================"
    echo "DONE."
    echo "  FedNoRo: ${FEDNORO_OUT}/  (summary.json + fednoro_full.csv)"
    echo "  非对称 : ${ASYM_OUT}/      (summary.json + asym_full.csv)"
    echo "============================================================"
    ;;
  *) echo "未知模式: ${MODE} (用 all | data | run | fednoro | asym | smoke)"; exit 1 ;;
esac
