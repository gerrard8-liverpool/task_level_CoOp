#!/usr/bin/env bash
set -euo pipefail

# Example usage:
# bash extract_task_features.sh \
#   /workspace/meta_prompt_1/third_party/CoOp \
#   /workspace/datasets \
#   /workspace/meta_prompt_1/third_party/CoOp/configs/datasets/oxford_pets.yaml \
#   OxfordPets \
#   RN50 \
#   /workspace/meta_prompt_1/outputs/task_features/oxford_pets_train.json

COOP_ROOT=${1:?"need coop root"}
DATA_ROOT=${2:?"need dataset root"}
DATASET_CFG=${3:?"need dataset config yaml"}
DATASET_NAME=${4:?"need dataset name"}
BACKBONE=${5:-RN50}
OUTPUT_JSON=${6:?"need output json"}

python /mnt/data/task_feature_extractor.py \
  --coop-root "$COOP_ROOT" \
  --root "$DATA_ROOT" \
  --dataset-config-file "$DATASET_CFG" \
  --dataset "$DATASET_NAME" \
  --backbone "$BACKBONE" \
  --split train_x \
  --text-template '{}' \
  --output "$OUTPUT_JSON"
