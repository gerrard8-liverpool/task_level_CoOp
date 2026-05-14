#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPU=0 bash scripts/backbone_dg/02_extract_backbone_task_features.sh rn101 caltech101 food101 sun397
#   GPU=0 bash scripts/backbone_dg/02_extract_backbone_task_features.sh vit_b16 caltech101 food101 sun397

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
SRC_ROOT=${SRC_ROOT:-$ROOT/src}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}
SEED=${SEED:-1}
SHOTS=${SHOTS:-16}
NUM_WORKERS=${NUM_WORKERS:-4}

BACKBONE_TAG=${1:?Usage: bash 02_extract_backbone_task_features.sh <rn101|vit_b16|vit_b32> [dataset ...]}
shift || true
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(caltech101 food101 sun397)
fi

case "$BACKBONE_TAG" in
  rn101)
    BACKBONE_NAME="RN101"
    CFG_TAG="rn101_ep50"
    BATCH_SIZE=${BATCH_SIZE:-64}
    ;;
  vit_b16)
    BACKBONE_NAME="ViT-B/16"
    CFG_TAG="vit_b16_ep50"
    BATCH_SIZE=${BATCH_SIZE:-32}
    ;;
  vit_b32)
    BACKBONE_NAME="ViT-B/32"
    CFG_TAG="vit_b32_ep50"
    BATCH_SIZE=${BATCH_SIZE:-48}
    ;;
  *)
    echo "[ERROR] Unknown BACKBONE_TAG=$BACKBONE_TAG. Use rn101, vit_b16, or vit_b32."
    exit 1
    ;;
esac

CFG_FILE="$COOP_ROOT/configs/trainers/CoOp/${CFG_TAG}.yaml"
if [ ! -f "$CFG_FILE" ]; then
  echo "[ERROR] Missing config: $CFG_FILE"
  echo "Run: bash scripts/backbone_dg/01_create_backbone_configs.sh"
  exit 1
fi

OUT_DIR="$ROOT/outputs/task_features_backbone/${BACKBONE_TAG}"
mkdir -p "$OUT_DIR"

export PYTHONPATH="$SRC_ROOT:$COOP_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

for DATASET in "${DATASETS[@]}"; do
  DATASET_CFG="$COOP_ROOT/configs/datasets/${DATASET}.yaml"
  OUT_JSON="$OUT_DIR/${DATASET}_train.json"

  if [ ! -f "$DATASET_CFG" ]; then
    echo "[ERROR] Missing dataset config: $DATASET_CFG"
    exit 1
  fi

  if [ -f "$OUT_JSON" ]; then
    echo "[SKIP] Existing feature: $OUT_JSON"
    continue
  fi

  echo "============================================================"
  echo "[TASK FEATURE] backbone=$BACKBONE_NAME tag=$BACKBONE_TAG dataset=$DATASET"
  echo "output=$OUT_JSON"
  echo "============================================================"

  python "$SRC_ROOT/meta_prompts/task_feature_extractor.py" \
    --coop-root "$COOP_ROOT" \
    --root "$DATA_ROOT" \
    --dataset-config-file "$DATASET_CFG" \
    --config-file "$CFG_FILE" \
    --backbone "$BACKBONE_NAME" \
    --split train_x \
    --num-shots "$SHOTS" \
    --subsample-classes all \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --output "$OUT_JSON"
done

find "$OUT_DIR" -maxdepth 1 -type f -name "*_train.json" -print | sort
