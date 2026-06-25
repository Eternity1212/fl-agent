#!/usr/bin/env bash
# =============================================================================
# 一键运行 agent (自适应编排) 实验
# =============================================================================
# 用法 (在已 clone 的仓库根目录, 已 export HF_TOKEN 或设 RETFOUND_CKPT_PATH):
#
#   ./run_agent.sh              # 先本地 smoke 验证, 再跑 GPU 全量 agent 矩阵
#   ./run_agent.sh smoke        # 只跑本地 smoke (CPU, ~1 分钟, 不需要 GPU/权重)
#   ./run_agent.sh full         # 跳过 smoke, 直接跑 GPU 全量矩阵
#
# 环境变量:
#   SKIP_SMOKE=1   跳过 smoke
#   OUT=...        全量结果输出目录 (默认 runs/paper_matrix/agent)
#
# 断点续跑: 每个 run 单独存盘, 中断后重跑自动跳过已完成的 run。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"

MODE="${1:-all}"
OUT="${OUT:-runs/paper_matrix/agent}"
SMOKE_OUT="runs/paper_matrix/agent_smoke"

run_smoke() {
  echo "==================================================================="
  echo " [1/3] 本地 smoke (CPU, TinyCNN, 子集) — 验证 agent 管线与机制"
  echo "==================================================================="
  python3 -m fed_agent.tools.run_paper_matrix \
    --matrix_yaml configs/paper_matrix_agent_smoke.yaml \
    --out_dir "${SMOKE_OUT}"
  python3 -m fed_agent.tools.summarize_agent "${SMOKE_OUT}/summary.json"
  echo ">>> smoke 完成。若上面 agent 的指标 >= fedavg, 说明机制有效, 可放心跑全量。"
}

run_full() {
  echo "==================================================================="
  echo " [2/3] GPU 全量 agent 矩阵 -> ${OUT}"
  echo "==================================================================="
  # 预检 RETFound 权重 (HF_TOKEN 或 RETFOUND_CKPT_PATH)
  python3 -m fed_agent.tools.check_env || {
    echo "!! 环境预检未通过: 请先 export HF_TOKEN=... 或 export RETFOUND_CKPT_PATH=/path/to/RETFound_mae_natureCFP.pth"
    exit 1
  }
  python3 -m fed_agent.tools.run_paper_matrix \
    --matrix_yaml configs/paper_matrix_agent.yaml \
    --out_dir "${OUT}"
  echo "==================================================================="
  echo " [3/3] 汇总对比 (fedavg vs robust vs agent)"
  echo "==================================================================="
  python3 -m fed_agent.tools.summarize_agent "${OUT}/summary.json"
}

case "${MODE}" in
  smoke) run_smoke ;;
  full)  run_full ;;
  all)
    [ "${SKIP_SMOKE:-0}" = "1" ] || run_smoke
    run_full
    ;;
  *) echo "未知模式: ${MODE} (用 smoke | full | all)"; exit 1 ;;
esac
