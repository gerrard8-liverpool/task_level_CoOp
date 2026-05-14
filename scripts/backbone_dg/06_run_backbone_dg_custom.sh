#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPU=2 bash scripts/backbone_dg/06_run_backbone_dg_custom.sh vit_b16 food101 eurosat oxford_pets sun397
#   GPU=2 bash scripts/backbone_dg/06_run_backbone_dg_custom.sh vit_b16 sun397 oxford_pets dtd food101

BACKBONE_TAG=${1:?Usage: bash 06_run_backbone_dg_custom.sh <rn101|vit_b16|vit_b32> <source> <target1> [target2 ...]}
SOURCE=${2:?source required}
shift 2

if [ "$#" -lt 1 ]; then
  echo "[ERROR] At least one target is required."
  exit 1
fi

TARGETS=("$@")

ROOT=${ROOT:-/workspace/meta_prompt_1}
GPU=${GPU:-0}

cd "$ROOT"

echo "============================================================"
echo "[CUSTOM BACKBONE DG]"
echo "BACKBONE_TAG=${BACKBONE_TAG}"
echo "SOURCE=${SOURCE}"
echo "TARGETS=${TARGETS[*]}"
echo "GPU=${GPU}"
echo "============================================================"

bash scripts/backbone_dg/01_create_backbone_configs.sh

# Build source feature with existing feature extraction script
GPU="$GPU" bash scripts/backbone_dg/02_extract_backbone_task_features.sh "$BACKBONE_TAG" "$SOURCE"

FEATURE_JSON="$ROOT/outputs/task_features_backbone/${BACKBONE_TAG}/${SOURCE}_train.json"

if [ ! -f "$FEATURE_JSON" ]; then
  echo "[ERROR] Feature not found after extraction: $FEATURE_JSON"
  exit 1
fi

for SEED in 1 2 3; do
  echo "============================================================"
  echo "[RUN COOP] backbone=${BACKBONE_TAG} source=${SOURCE} seed=${SEED}"
  echo "============================================================"

  GPU="$GPU" \
  bash scripts/backbone_dg/03_run_backbone_xd_one.sh \
    coop "$BACKBONE_TAG" "$SOURCE" "$SEED" "${TARGETS[@]}"

  echo "============================================================"
  echo "[RUN SAFE] backbone=${BACKBONE_TAG} source=${SOURCE} seed=${SEED}"
  echo "============================================================"

  GPU="$GPU" \
  SOURCE_FEATURE_JSON="$FEATURE_JSON" \
  bash scripts/backbone_dg/03_run_backbone_xd_one.sh \
    safe "$BACKBONE_TAG" "$SOURCE" "$SEED" "${TARGETS[@]}"
done

mkdir -p outputs/xd_main_tables

python scripts/backbone_dg/06_summarize_backbone_dg.py \
  "$BACKBONE_TAG" \
  --sources "$SOURCE" \
  --targets "${TARGETS[@]}" \
  | tee "outputs/xd_main_tables/xd_${BACKBONE_TAG}_${SOURCE}_custom_coop_safe.md"

echo "[DONE] outputs/xd_main_tables/xd_${BACKBONE_TAG}_${SOURCE}_custom_coop_safe.md"
