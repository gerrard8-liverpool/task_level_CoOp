#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
SRC_ROOT=${SRC_ROOT:-$ROOT/src}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

VARIANT=${1:?Usage: bash run_priorres_residual_indomain.sh <safe|legacy> <dataset> <seed>}
DATASET=${2:?Usage: bash run_priorres_residual_indomain.sh <safe|legacy> <dataset> <seed>}
SEED=${3:?Usage: bash run_priorres_residual_indomain.sh <safe|legacy> <dataset> <seed>}

GPU=${GPU:-0}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CTX_POS=${CTX_POS:-end}
CSC=${CSC:-False}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoOp/rn50_ep50.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_ep50}
TRAIN_BS=${TRAIN_BS:-16}
TEST_BS=${TEST_BS:-64}

case "$VARIANT" in
  safe)
    LEGACY_FLAG=False
    VARIANT_TAG=safe
    ;;
  legacy)
    LEGACY_FLAG=True
    VARIANT_TAG=legacy
    ;;
  *)
    echo "[ERROR] VARIANT must be safe or legacy, got: $VARIANT"
    exit 1
    ;;
esac

DATASET_CFG=${DATASET_CFG:-$COOP_ROOT/configs/datasets/${DATASET}.yaml}
TASK_FEATURE_JSON=${TASK_FEATURE_JSON:-$ROOT/outputs/task_features/${DATASET}_train.json}
OUT_DIR="$ROOT/outputs/ablations/residual_formula/runs/indomain/${DATASET}/${VARIANT_TAG}/shots_${SHOTS}/CoOpPriorRes/${TRAIN_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}"

export PYTHONPATH="$SRC_ROOT:$COOP_ROOT:${PYTHONPATH:-}"
export USE_LEGACY_RESIDUAL="$LEGACY_FLAG"

cd "$COOP_ROOT"

echo "============================================================"
echo "[INDOMAIN PRIORRES RESIDUAL FORMULA ABLATION]"
echo "variant   = ${VARIANT_TAG}"
echo "legacy    = ${USE_LEGACY_RESIDUAL}"
echo "dataset   = ${DATASET}"
echo "seed      = ${SEED}"
echo "out       = ${OUT_DIR}"
echo "feature   = ${TASK_FEATURE_JSON}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoOpPriorRes \
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
  DATASET.SUBSAMPLE_CLASSES all \
  TRAINER.COOP.TASK_FEAT_PATH "$TASK_FEATURE_JSON" \
  TRAINER.COOP.TASK_FEAT_MODE transformed \
  TRAINER.COOP.META_HIDDEN_DIM 64 \
  TRAINER.COOP.META_KMAX 16 \
  TRAINER.COOP.GATE_TEMPERATURE 1.0 \
  TRAINER.COOP.INIT_GATE_BIAS 4.0 \
  TRAINER.COOP.USE_CONTEXT_GATING True \
  TRAINER.COOP.USE_B False \
  TRAINER.COOP.B_LOSS_WEIGHT 1.0 \
  TRAINER.COOP.WARMUP_EPOCHS 5 \
  TRAINER.COOP.RAMP_EPOCHS 10 \
  TRAINER.COOP.LAMBDA_MAX 1.0 \
  TRAINER.COOP.ALTERNATE_OPT False \
  TRAINER.COOP.META_LR_RATIO 0.3 \
  TRAINER.COOP.DELTA_A_REG 1e-4 \
  TRAINER.COOP.DELTA_B_REG 1e-4
