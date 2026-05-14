#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

DATASET=${1:?Usage: GPU=0 bash 00_extract_b2n_base_features.sh <dataset> [seed]}
SEED=${2:-1}
GPU=${GPU:-0}

OUT_DIR=${OUT_DIR:-$ROOT/outputs/task_features_cocoop_b2n_base/rn50}
OUT_JSON="$OUT_DIR/${DATASET}_seed${SEED}_base_train.json"

mkdir -p "$OUT_DIR"

cd "$ROOT"

echo "============================================================"
echo "[Extract B2N base-only task feature]"
echo "dataset = $DATASET"
echo "seed    = $SEED"
echo "out     = $OUT_JSON"
echo "============================================================"

if [ "${FORCE_CPU_FEATURE:-0}" = "1" ]; then
  echo "[INFO] FORCE_CPU_FEATURE=1, extracting task feature on CPU"
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

PYTHONFAULTHANDLER=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
python "$ROOT/src/meta_prompts/task_feature_extractor.py" \
  --coop-root "$COOP_ROOT" \
  --root "$DATA_ROOT" \
  --dataset-config-file "$COOP_ROOT/configs/datasets/${DATASET}.yaml" \
  --config-file "$COOP_ROOT/configs/trainers/CoCoOp/rn50_c4_ep10_batch1.yaml" \
  --backbone RN50 \
  --split train_x \
  --num-shots 16 \
  --subsample-classes base \
  --seed "$SEED" \
  --output "$OUT_JSON" \
  DATALOADER.NUM_WORKERS 0

echo "[DONE] $OUT_JSON"
