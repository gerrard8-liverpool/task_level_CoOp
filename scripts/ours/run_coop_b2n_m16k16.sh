#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

DATASET=${1:?Usage: bash run_coop_b2n_m16k16.sh <dataset> <seed> [shots] [nctx]}
SEED=${2:?Usage: bash run_coop_b2n_m16k16.sh <dataset> <seed> [shots] [nctx]}
SHOTS=${3:-16}
NCTX=${4:-16}

GPU=${GPU:-1}
CTX_POS=${CTX_POS:-end}
CSC=${CSC:-False}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoOp/rn50_ep50.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_ep50}
LOAD_EPOCH=${LOAD_EPOCH:-50}

DATASET_CFG=${DATASET_CFG:-$COOP_ROOT/configs/datasets/${DATASET}.yaml}

COMMON="${DATASET}/shots_${SHOTS}/CoOp/${TRAIN_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}"
TRAIN_DIR="$COOP_ROOT/output/base2new/train_base/${COMMON}"

cd "$COOP_ROOT"

echo "============================================================"
echo "[CoOp B2N TRAIN BASE] dataset=${DATASET} seed=${SEED}"
echo "[OUT] ${TRAIN_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoOp \
  --dataset-config-file "$DATASET_CFG" \
  --config-file "$TRAIN_CFG" \
  --output-dir "$TRAIN_DIR" \
  --seed "$SEED" \
  TRAINER.COOP.N_CTX "$NCTX" \
  TRAINER.COOP.CSC "$CSC" \
  TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
  DATASET.NUM_SHOTS "$SHOTS" \
  DATASET.SUBSAMPLE_CLASSES base

for SUB in base new; do
  TEST_DIR="$COOP_ROOT/output/base2new/test_${SUB}/${COMMON}"

  echo "============================================================"
  echo "[CoOp B2N TEST ${SUB}] dataset=${DATASET} seed=${SEED}"
  echo "[MODEL] ${TRAIN_DIR}"
  echo "[OUT] ${TEST_DIR}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "$DATA_ROOT" \
    --trainer CoOp \
    --dataset-config-file "$DATASET_CFG" \
    --config-file "$TRAIN_CFG" \
    --output-dir "$TEST_DIR" \
    --model-dir "$TRAIN_DIR" \
    --load-epoch "$LOAD_EPOCH" \
    --eval-only \
    --seed "$SEED" \
    TRAINER.COOP.N_CTX "$NCTX" \
    TRAINER.COOP.CSC "$CSC" \
    TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
    DATASET.NUM_SHOTS "$SHOTS" \
    DATASET.SUBSAMPLE_CLASSES "$SUB"
done

echo "[DONE] CoOp B2N dataset=${DATASET} seed=${SEED}"
