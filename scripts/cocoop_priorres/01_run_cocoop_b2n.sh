#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

DATASET=${1:?Usage: GPU=0 bash 01_run_cocoop_b2n.sh <dataset> <seed>}
SEED=${2:?Usage: GPU=0 bash 01_run_cocoop_b2n.sh <dataset> <seed>}

GPU=${GPU:-0}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-4}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoCoOp/rn50_c4_ep10_batch1.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_c4_ep10_batch1}
LOAD_EPOCH=${LOAD_EPOCH:-10}

DATASET_CFG="$COOP_ROOT/configs/datasets/${DATASET}.yaml"
COMMON="${DATASET}/shots_${SHOTS}/CoCoOp/${TRAIN_CFG_TAG}/nctx${NCTX}_ctxinit_a_photo_of_a/seed${SEED}"
TRAIN_DIR="$COOP_ROOT/output_cocoop_priorres/base2new/train_base/${COMMON}"

cd "$COOP_ROOT"

echo "============================================================"
echo "[CoCoOp B2N TRAIN BASE]"
echo "dataset = $DATASET"
echo "seed    = $SEED"
echo "out     = $TRAIN_DIR"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoCoOp \
  --dataset-config-file "$DATASET_CFG" \
  --config-file "$TRAIN_CFG" \
  --output-dir "$TRAIN_DIR" \
  --seed "$SEED" \
  DATASET.NUM_SHOTS "$SHOTS" \
  DATASET.SUBSAMPLE_CLASSES base \
  TRAINER.COCOOP.N_CTX "$NCTX" \
  TRAINER.COCOOP.CTX_INIT a_photo_of_a

for SUB in base new; do
  TEST_DIR="$COOP_ROOT/output_cocoop_priorres/base2new/test_${SUB}/${COMMON}"

  echo "============================================================"
  echo "[CoCoOp B2N TEST $SUB]"
  echo "dataset = $DATASET"
  echo "seed    = $SEED"
  echo "model   = $TRAIN_DIR"
  echo "out     = $TEST_DIR"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "$DATA_ROOT" \
    --trainer CoCoOp \
    --dataset-config-file "$DATASET_CFG" \
    --config-file "$TRAIN_CFG" \
    --output-dir "$TEST_DIR" \
    --model-dir "$TRAIN_DIR" \
    --load-epoch "$LOAD_EPOCH" \
    --eval-only \
    --seed "$SEED" \
    DATASET.NUM_SHOTS "$SHOTS" \
    DATASET.SUBSAMPLE_CLASSES "$SUB" \
    TRAINER.COCOOP.N_CTX "$NCTX" \
    TRAINER.COCOOP.CTX_INIT a_photo_of_a
done
