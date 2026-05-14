#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

DATASET=${1:?Usage: bash run_coop_residual_indomain.sh <dataset> <seed>}
SEED=${2:?Usage: bash run_coop_residual_indomain.sh <dataset> <seed>}

GPU=${GPU:-0}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CTX_POS=${CTX_POS:-end}
CSC=${CSC:-False}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoOp/rn50_ep50.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_ep50}
TRAIN_BS=${TRAIN_BS:-16}
TEST_BS=${TEST_BS:-64}

DATASET_CFG=${DATASET_CFG:-$COOP_ROOT/configs/datasets/${DATASET}.yaml}
OUT_DIR="$ROOT/outputs/ablations/residual_formula/runs/indomain/${DATASET}/coop/shots_${SHOTS}/CoOp/${TRAIN_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}"

cd "$COOP_ROOT"

echo "============================================================"
echo "[INDOMAIN COOP BASELINE FOR RESIDUAL FORMULA ABLATION]"
echo "dataset = ${DATASET}"
echo "seed    = ${SEED}"
echo "out     = ${OUT_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoOp \
  --dataset-config-file "$DATASET_CFG" \
  --config-file "$TRAIN_CFG" \
  --output-dir "$OUT_DIR" \
  --seed "$SEED" \
  TRAINER.COOP.N_CTX "$NCTX" \
  TRAINER.COOP.CSC "$CSC" \
  TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
  TRAINER.COOP.PREC fp32 \
  DATASET.NUM_SHOTS "$SHOTS" \
  DATALOADER.TRAIN_X.BATCH_SIZE "$TRAIN_BS" \
  DATALOADER.TEST.BATCH_SIZE "$TEST_BS" \
  DATASET.SUBSAMPLE_CLASSES all
