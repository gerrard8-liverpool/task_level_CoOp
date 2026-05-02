#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

SOURCE=${1:?Usage: bash run_coop_xd_m16k16.sh <source_dataset> <seed> <target1> [target2 ...]}
SEED=${2:?Usage: bash run_coop_xd_m16k16.sh <source_dataset> <seed> <target1> [target2 ...]}
shift 2
TARGETS=("$@")

if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "[ERROR] Please provide at least one target dataset."
  exit 1
fi

GPU=${GPU:-1}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CTX_POS=${CTX_POS:-end}
CSC=${CSC:-False}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoOp/rn50_ep50.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_ep50}
LOAD_EPOCH=${LOAD_EPOCH:-50}
PREC=${PREC:-fp16}
TRAIN_BS=${TRAIN_BS:-16}
TEST_BS=${TEST_BS:-64}

SOURCE_CFG=${SOURCE_CFG:-$COOP_ROOT/configs/datasets/${SOURCE}.yaml}

COMMON="source_${SOURCE}/shots_${SHOTS}/CoOp/${TRAIN_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}"
TRAIN_DIR="$COOP_ROOT/output/xd/train/${COMMON}"

cd "$COOP_ROOT"

echo "============================================================"
echo "[XD CoOp TRAIN SOURCE]"
echo "source = ${SOURCE}"
echo "seed   = ${SEED}"
echo "out    = ${TRAIN_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoOp \
  --dataset-config-file "$SOURCE_CFG" \
  --config-file "$TRAIN_CFG" \
  --output-dir "$TRAIN_DIR" \
  --seed "$SEED" \
  TRAINER.COOP.N_CTX "$NCTX" \
  TRAINER.COOP.CSC "$CSC" \
  TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
  TRAINER.COOP.PREC fp32 \
  DATASET.NUM_SHOTS "$SHOTS" \
  DATALOADER.TRAIN_X.BATCH_SIZE "$TRAIN_BS" \
  DATALOADER.TEST.BATCH_SIZE "$TEST_BS" \
  DATASET.SUBSAMPLE_CLASSES all

for TARGET in "${TARGETS[@]}"; do
  TARGET_CFG="$COOP_ROOT/configs/datasets/${TARGET}.yaml"
  TEST_DIR="$COOP_ROOT/output/xd/test/${TARGET}/${COMMON}"

  echo "============================================================"
  echo "[XD CoOp TEST TARGET]"
  echo "source = ${SOURCE}"
  echo "target = ${TARGET}"
  echo "seed   = ${SEED}"
  echo "model  = ${TRAIN_DIR}"
  echo "out    = ${TEST_DIR}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "$DATA_ROOT" \
    --trainer CoOp \
    --dataset-config-file "$TARGET_CFG" \
    --config-file "$TRAIN_CFG" \
    --output-dir "$TEST_DIR" \
    --model-dir "$TRAIN_DIR" \
    --load-epoch "$LOAD_EPOCH" \
    --eval-only \
    --seed "$SEED" \
    TRAINER.COOP.N_CTX "$NCTX" \
    TRAINER.COOP.CSC "$CSC" \
    TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
  TRAINER.COOP.PREC fp32 \
    DATASET.NUM_SHOTS "$SHOTS" \
  DATALOADER.TRAIN_X.BATCH_SIZE "$TRAIN_BS" \
  DATALOADER.TEST.BATCH_SIZE "$TEST_BS" \
    DATASET.SUBSAMPLE_CLASSES all
done

echo "[DONE] XD CoOp source=${SOURCE} seed=${SEED}"
