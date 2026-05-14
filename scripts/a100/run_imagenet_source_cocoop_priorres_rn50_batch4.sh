#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/CoOp_clean:$PROJECT_ROOT/third_party/Dassl.pytorch:${PYTHONPATH:-}"
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

TRAIN_CFG="configs/trainers/CoCoOp/rn50_c4_ep10_batch4_a100.yaml"
TRAIN_CFG_TAG="rn50_c4_ep10_batch4_a100"
FEATURE_JSON="$PROJECT_ROOT/outputs/task_features/imagenet_train_sample32.json"

if [ ! -f "$FEATURE_JSON" ]; then
  echo "[ERROR] Missing ImageNet source feature: $FEATURE_JSON"
  exit 1
fi

echo "============================================================"
echo "[ImageNet-source CoCoOp / CoCoOpPriorRes DG]"
echo "SOURCE=$SOURCE"
echo "TARGETS=${TARGETS[*]}"
echo "TRAIN_CFG=$TRAIN_CFG"
echo "TRAIN_CFG_TAG=$TRAIN_CFG_TAG"
echo "FEATURE_JSON=$FEATURE_JSON"
echo "GPU=$GPU"
echo "============================================================"

for SEED in 1 2 3; do
  echo "============================================================"
  echo "[Seed $SEED] CoCoOp train"
  echo "============================================================"

  GPU="$GPU" \
  TRAIN_CFG="$TRAIN_CFG" \
  TRAIN_CFG_TAG="$TRAIN_CFG_TAG" \
  bash scripts/cocoop_priorres/05a_train_cocoop_xd_only.sh \
    "$SOURCE" "$SEED" \
    2>&1 | tee "logs/a100_cocoop_imagenet_dg/cocoop_imagenet_train_seed${SEED}.log"

  echo "============================================================"
  echo "[Seed $SEED] CoCoOp test"
  echo "============================================================"

  GPU="$GPU" \
  TRAIN_CFG="$TRAIN_CFG" \
  TRAIN_CFG_TAG="$TRAIN_CFG_TAG" \
  bash scripts/cocoop_priorres/05b_test_cocoop_xd_only.sh \
    "$SOURCE" "$SEED" "${TARGETS[@]}" \
    2>&1 | tee "logs/a100_cocoop_imagenet_dg/cocoop_imagenet_test_seed${SEED}.log"

  echo "============================================================"
  echo "[Seed $SEED] CoCoOpPriorRes train"
  echo "============================================================"

  GPU="$GPU" \
  TRAIN_CFG="$TRAIN_CFG" \
  TRAIN_CFG_TAG="$TRAIN_CFG_TAG" \
  SOURCE_FEATURE_JSON="$FEATURE_JSON" \
  bash scripts/cocoop_priorres/06a_train_cocoop_priorres_xd_only.sh \
    "$SOURCE" "$SEED" \
    2>&1 | tee "logs/a100_cocoop_imagenet_dg/priorres_imagenet_train_seed${SEED}.log"

  echo "============================================================"
  echo "[Seed $SEED] CoCoOpPriorRes test"
  echo "============================================================"

  GPU="$GPU" \
  TRAIN_CFG="$TRAIN_CFG" \
  TRAIN_CFG_TAG="$TRAIN_CFG_TAG" \
  SOURCE_FEATURE_JSON="$FEATURE_JSON" \
  bash scripts/cocoop_priorres/06b_test_cocoop_priorres_xd_only.sh \
    "$SOURCE" "$SEED" "${TARGETS[@]}" \
    2>&1 | tee "logs/a100_cocoop_imagenet_dg/priorres_imagenet_test_seed${SEED}.log"
done

echo "============================================================"
echo "[DONE] ImageNet-source CoCoOp / CoCoOpPriorRes DG"
echo "============================================================"
