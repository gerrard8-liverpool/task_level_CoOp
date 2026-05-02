#!/usr/bin/env bash
set -euo pipefail

cd /workspace/meta_prompt_1

SOURCE=${1:?Usage: bash scripts/ours/run_xd_pair_compare.sh <source_dataset> <gpu_priorres> <gpu_coop> <target1> [target2 ...]}
GPU_PRIORRES=${2:?Usage: bash scripts/ours/run_xd_pair_compare.sh <source_dataset> <gpu_priorres> <gpu_coop> <target1> [target2 ...]}
GPU_COOP=${3:?Usage: bash scripts/ours/run_xd_pair_compare.sh <source_dataset> <gpu_priorres> <gpu_coop> <target1> [target2 ...]}
shift 3

TARGETS=("$@")

if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "[ERROR] Please provide at least one target dataset."
  exit 1
fi

SEEDS=(1 2 3)

echo "============================================================"
echo "[XD PAIR COMPARE]"
echo "source       = ${SOURCE}"
echo "gpu_priorres = ${GPU_PRIORRES}"
echo "gpu_coop     = ${GPU_COOP}"
echo "targets      = ${TARGETS[*]}"
echo "seeds        = ${SEEDS[*]}"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
  echo
  echo "============================================================"
  echo "[PriorRes] source=${SOURCE} seed=${SEED}"
  echo "============================================================"

  GPU=${GPU_PRIORRES} TEST_FEATURE_MODE=source \
  bash scripts/ours/run_priorres_xd_safe_noalt.sh \
  "${SOURCE}" "${SEED}" "${TARGETS[@]}"

  echo
  echo "============================================================"
  echo "[CoOp] source=${SOURCE} seed=${SEED}"
  echo "============================================================"

  GPU=${GPU_COOP} \
  bash scripts/ours/run_coop_xd_m16k16.sh \
  "${SOURCE}" "${SEED}" "${TARGETS[@]}"
done

OUT_MD="outputs/xd_${SOURCE}_compare_auto.md"

echo
echo "============================================================"
echo "[SUMMARY] ${OUT_MD}"
echo "============================================================"

python scripts/ours/summarize_xd_compare.py "${SOURCE}" "${TARGETS[@]}" > "${OUT_MD}"

cat "${OUT_MD}"

echo
echo "============================================================"
echo "[DONE] XD pair compare finished."
echo "Output: ${OUT_MD}"
echo "============================================================"
