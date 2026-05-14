#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

SOURCE=${1:?Usage: GPU=0 bash 05_run_cocoop_xd.sh <source> <seed> <target1> [target2 ...]}
SEED=${2:?Usage: GPU=0 bash 05_run_cocoop_xd.sh <source> <seed> <target1> [target2 ...]}
shift 2
TARGETS=("$@")

GPU=${GPU:-0}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-4}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoCoOp/rn50_c4_ep10_batch1.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_c4_ep10_batch1}
LOAD_EPOCH=${LOAD_EPOCH:-10}

SOURCE_CFG="$COOP_ROOT/configs/datasets/${SOURCE}.yaml"
COMMON="source_${SOURCE}/shots_${SHOTS}/CoCoOp/${TRAIN_CFG_TAG}/nctx${NCTX}_ctxinit_a_photo_of_a/seed${SEED}"
TRAIN_DIR="$COOP_ROOT/output_cocoop_priorres/xd/train/${COMMON}"

cd "$COOP_ROOT"

echo "============================================================"
echo "[CoCoOp XD TRAIN SOURCE]"
echo "source = $SOURCE"
echo "seed   = $SEED"
echo "out    = $TRAIN_DIR"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoCoOp \
  --dataset-config-file "$SOURCE_CFG" \
  --config-file "$TRAIN_CFG" \
  --output-dir "$TRAIN_DIR" \
  --seed "$SEED" \
  DATASET.NUM_SHOTS "$SHOTS" \
  DATASET.SUBSAMPLE_CLASSES all \
  TRAINER.COCOOP.N_CTX "$NCTX" \
  TRAINER.COCOOP.CTX_INIT a_photo_of_a

for TARGET in "${TARGETS[@]}"; do
  TARGET_CFG="$COOP_ROOT/configs/datasets/${TARGET}.yaml"
  TEST_DIR="$COOP_ROOT/output_cocoop_priorres/xd/test/${TARGET}/${COMMON}"

  echo "============================================================"
  echo "[CoCoOp XD TEST TARGET]"
  echo "source = $SOURCE"
  echo "target = $TARGET"
  echo "seed   = $SEED"
  echo "model  = $TRAIN_DIR"
  echo "out    = $TEST_DIR"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "$DATA_ROOT" \
    --trainer CoCoOp \
    --dataset-config-file "$TARGET_CFG" \
    --config-file "$TRAIN_CFG" \
    --output-dir "$TEST_DIR" \
    --model-dir "$TRAIN_DIR" \
    --load-epoch "$LOAD_EPOCH" \
    --eval-only \
    --seed "$SEED" \
    DATASET.NUM_SHOTS "$SHOTS" \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.COCOOP.N_CTX "$NCTX" \
    TRAINER.COCOOP.CTX_INIT a_photo_of_a
done
