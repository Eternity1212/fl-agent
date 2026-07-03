#!/usr/bin/env bash
# =============================================================================
# 一键运行 ODIR 第二数据集 —— agent 贡献跨数据集复现 (GPU)
# =============================================================================
# 目的: 在第二个多标签眼底数据集 ODIR(8类)上复现 RFMiD 的核心二分模式——
#   "稀释型聚合(FedAvg/floor)非IID F1 全崩 vs 重加权家族(agent/ccr)在多数划分恢复"。
#   若在 ODIR 重现, 这个机制就从单数据集观察升级为跨数据集规律。
#
# 一条龙: 装依赖 -> 下载ODIR -> 生成split -> 跑23个run -> 汇总
#
# 用法 (在已 clone 仓库根目录):
#   git fetch origin && git checkout feat/agent-supplement && git pull
#   export HF_TOKEN=hf_xxx            # 或 export RETFOUND_CKPT_PATH=/path/RETFound_mae_natureCFP.pth
#   ./run_odir.sh                     # 全套: 数据+split+矩阵+汇总
#
# 子命令:
#   ./run_odir.sh data               # 只做数据下载 + split 生成 (可用 CPU 与 GPU 主线并行)
#   ./run_odir.sh run                # 跳过数据准备, 只跑矩阵 + 汇总 (数据已就绪时)
#
# 环境变量:
#   ODIR_DATA_DIR=...    默认 data/raw/odir
#   SEEDS="0 1 2"        生成 split 的种子
#   OUT=...              结果输出目录 (默认 runs/paper_matrix/agent_odir)
#   SKIP_INSTALL=1       跳过 pip install
#
# 断点续跑: 每个 run 单独存盘, 中断重跑自动跳过已完成的 run。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"

MODE="${1:-all}"
DATA_DIR="${ODIR_DATA_DIR:-data/raw/odir}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-runs/paper_matrix/agent_odir}"
MATRIX="configs/paper_matrix_agent_odir.yaml"

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
  echo " 下载 ODIR -> ${DATA_DIR}  (已存在文件自动跳过)"
  echo "==================================================================="
  python3 -m fed_agent.tools.export_odir --source hf --out_dir "${DATA_DIR}"

  echo "==================================================================="
  echo " 生成联邦 split (IID / Dirichlet a=0.1,0.5,1.0), seeds: ${SEEDS}"
  echo "==================================================================="
  for seed in ${SEEDS}; do
    python3 -m fed_agent.tools.build_splits \
      --labels_csv "${DATA_DIR}/train/labels.csv" \
      --out_dir configs/splits/generated_odir \
      --seed "${seed}" \
      --n_clients 4 \
      --alphas 0.1 0.5 1.0
  done
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
  echo " 跑 ODIR 复现矩阵 (23 run) -> ${OUT}"
  echo "==================================================================="
  python3 -m fed_agent.tools.run_paper_matrix \
    --matrix_yaml "${MATRIX}" \
    --out_dir "${OUT}"
  echo "==================================================================="
  echo " 汇总 (重点看: 非IID 下 fedavg/floor 是否全崩, agent/ccr 是否恢复)"
  echo "==================================================================="
  python3 -m fed_agent.tools.summarize_agent "${OUT}/summary.json"
}

case "${MODE}" in
  data) install_deps; prep_data ;;
  run)  run_matrix ;;
  all)
    install_deps
    prep_data
    run_matrix
    echo ""
    echo "============================================================"
    echo "DONE. 结果在: ${OUT}/  (summary.json + 各 run json)"
    echo "============================================================"
    ;;
  *) echo "未知模式: ${MODE} (用 all | data | run)"; exit 1 ;;
esac
