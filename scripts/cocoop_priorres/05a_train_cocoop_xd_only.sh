#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

SOURCE=${1:?Usage: GPU=0 bash 05a_train_cocoop_xd_only.sh <source> <seed>}
SEED=${2:?Usage: GPU=0 bash 05a_train_cocoop_xd_only.sh <source> <seed>}

GPU=${GPU:-0}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-4}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoCoOp/rn50_c4_ep10_batch1.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_c4_ep10_batch1}

SOURCE_CFG="$COOP_ROOT/configs/datasets/${SOURCE}.yaml"
COMMON="source_${SOURCE}/shots_${SHOTS}/CoCoOp/${TRAIN_CFG_TAG}/nctx${NCTX}_ctxinit_a_photo_of_a/seed${SEED}"
TRAIN_DIR="$COOP_ROOT/output_cocoop_priorres/xd/train/${COMMON}"

if [ -f "$TRAIN_DIR/prompt_learner/model.pth.tar-10" ]; then
  echo "[SKIP] CoCoOp train done: $TRAIN_DIR"
  exit 0
fi

cd "$COOP_ROOT"

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
