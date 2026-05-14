#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/CoOp_clean:$PROJECT_ROOT/third_party/Dassl.pytorch:${PYTHONPATH:-}"
export UCX_TLS=tcp,self
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4
export PYTHONFAULTHANDLER=1

SOURCE="imagenet"
TARGETS=(
  caltech101
  oxford_pets
  dtd
  eurosat
  food101
  oxford_flowers
  stanford_cars
  fgvc_aircraft
  ucf101
  sun397
)

FEATURE_JSON="$PROJECT_ROOT/outputs/task_features/imagenet_train_sample32.json"

if [ ! -f "$FEATURE_JSON" ]; then
  echo "[ERROR] Missing ImageNet task feature: $FEATURE_JSON"
  echo "You need imagenet_train_sample32.json for Safe PriorRes."
  exit 1
fi

echo "============================================================"
echo "[ImageNet-source DG: RN50]"
echo "SOURCE=$SOURCE"
echo "TARGETS=${TARGETS[*]}"
echo "FEATURE_JSON=$FEATURE_JSON"
echo "GPU=$GPU"
echo "============================================================"

for SEED in 1 2 3; do
  echo "============================================================"
  echo "[CoOp] ImageNet source, seed=$SEED"
  echo "============================================================"

  GPU="$GPU" \
  bash scripts/ours/run_coop_xd_m16k16.sh \
    "$SOURCE" "$SEED" "${TARGETS[@]}" \
    2>&1 | tee "logs/a100_imagenet_dg/coop_imagenet_seed${SEED}.log"

  echo "============================================================"
  echo "[Safe PriorRes] ImageNet source, seed=$SEED"
  echo "============================================================"

  GPU="$GPU" \
  SOURCE_FEATURE_JSON="$FEATURE_JSON" \
  TEST_FEATURE_MODE=source \
  bash scripts/ours/run_priorres_xd_safe_noalt.sh \
    "$SOURCE" "$SEED" "${TARGETS[@]}" \
    2>&1 | tee "logs/a100_imagenet_dg/safe_imagenet_seed${SEED}.log"
done

echo "============================================================"
echo "[Summarize ImageNet-source DG]"
echo "============================================================"

python scripts/ours/summarize_xd_multisource_compare.py \
  imagenet \
  | tee outputs/xd_main_tables/xd_imagenet_source_coop_safe_rn50.md

echo "[DONE] outputs/xd_main_tables/xd_imagenet_source_coop_safe_rn50.md"
