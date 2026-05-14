#!/usr/bin/env bash
set -euo pipefail

# ===== User settings =====
GPU_ID=${GPU_ID:-0}
MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-3000}
CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC:-300}
STABLE_REQUIRED=${STABLE_REQUIRED:-3}

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
CONDA_ENV=${CONDA_ENV:-meta_prompt}

RUN_SCRIPT="${PROJECT_ROOT}/scripts/a100/run_imagenet_source_dg_rn50.sh"
LOG_DIR="${PROJECT_ROOT}/logs/a100_imagenet_dg"
WAIT_LOG="${LOG_DIR}/wait_gpu_then_run.log"
RUN_LOG="${LOG_DIR}/imagenet_source_dg_auto_run.log"
LOCK_FILE="${LOG_DIR}/imagenet_dg_started.lock"

mkdir -p "$LOG_DIR"

echo "============================================================" | tee -a "$WAIT_LOG"
echo "[WAIT GPU THEN RUN IMAGENET DG]" | tee -a "$WAIT_LOG"
echo "Time: $(date)" | tee -a "$WAIT_LOG"
echo "GPU_ID=${GPU_ID}" | tee -a "$WAIT_LOG"
echo "MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB}" | tee -a "$WAIT_LOG"
echo "CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC}" | tee -a "$WAIT_LOG"
echo "STABLE_REQUIRED=${STABLE_REQUIRED}" | tee -a "$WAIT_LOG"
echo "PROJECT_ROOT=${PROJECT_ROOT}" | tee -a "$WAIT_LOG"
echo "DATA_ROOT=${DATA_ROOT}" | tee -a "$WAIT_LOG"
echo "RUN_SCRIPT=${RUN_SCRIPT}" | tee -a "$WAIT_LOG"
echo "============================================================" | tee -a "$WAIT_LOG"

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "[ERROR] Missing run script: $RUN_SCRIPT" | tee -a "$WAIT_LOG"
  exit 1
fi

if [ -f "$LOCK_FILE" ]; then
  echo "[ERROR] Lock file exists, maybe the job was already started:" | tee -a "$WAIT_LOG"
  echo "$LOCK_FILE" | tee -a "$WAIT_LOG"
  echo "If you are sure you want to rerun, remove it manually:" | tee -a "$WAIT_LOG"
  echo "rm -f $LOCK_FILE" | tee -a "$WAIT_LOG"
  exit 1
fi

stable_count=0

while true; do
  timestamp=$(date "+%Y-%m-%d %H:%M:%S")

  mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
  util_gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')

  proc_count=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | grep -v "^$" | wc -l || true)

  echo "[$timestamp] GPU=${GPU_ID} mem_used=${mem_used}MiB util=${util_gpu}% proc_count=${proc_count} stable=${stable_count}/${STABLE_REQUIRED}" | tee -a "$WAIT_LOG"

  if [ "$mem_used" -le "$MEM_THRESHOLD_MB" ] && [ "$proc_count" -eq 0 ]; then
    stable_count=$((stable_count + 1))
  else
    stable_count=0
  fi

  if [ "$stable_count" -ge "$STABLE_REQUIRED" ]; then
    echo "[$timestamp] GPU is idle enough. Starting ImageNet-source DG..." | tee -a "$WAIT_LOG"
    touch "$LOCK_FILE"
    break
  fi

  sleep "$CHECK_INTERVAL_SEC"
done

cd "$PROJECT_ROOT"

# Activate conda inside non-interactive shell
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "[ERROR] conda command not found." | tee -a "$WAIT_LOG"
  exit 1
fi

export PROJECT_ROOT="$PROJECT_ROOT"
export DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/CoOp_clean:$PROJECT_ROOT/third_party/Dassl.pytorch:${PYTHONPATH:-}"
export UCX_TLS=tcp,self
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4
export PYTHONFAULTHANDLER=1

echo "============================================================" | tee -a "$RUN_LOG"
echo "[START IMAGENET-SOURCE DG]" | tee -a "$RUN_LOG"
echo "Time: $(date)" | tee -a "$RUN_LOG"
echo "GPU=${GPU_ID}" | tee -a "$RUN_LOG"
echo "============================================================" | tee -a "$RUN_LOG"

GPU="$GPU_ID" \
PROJECT_ROOT="$PROJECT_ROOT" \
DATA_ROOT="$DATA_ROOT" \
bash "$RUN_SCRIPT" 2>&1 | tee -a "$RUN_LOG"

echo "============================================================" | tee -a "$RUN_LOG"
echo "[FINISHED IMAGENET-SOURCE DG]" | tee -a "$RUN_LOG"
echo "Time: $(date)" | tee -a "$RUN_LOG"
echo "============================================================" | tee -a "$RUN_LOG"
