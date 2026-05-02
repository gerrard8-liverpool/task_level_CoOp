#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
SRC_ROOT=${SRC_ROOT:-$ROOT/src}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

DATASET=${1:?Usage: bash run_priorres_b2n_safe_noalt.sh <dataset> <seed> [shots] [nctx]}
SEED=${2:?Usage: bash run_priorres_b2n_safe_noalt.sh <dataset> <seed> [shots] [nctx]}
SHOTS=${3:-16}
NCTX=${4:-16}

GPU=${GPU:-1}
CTX_POS=${CTX_POS:-end}
CSC=${CSC:-False}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoOp/rn50_ep50.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_ep50}
LOAD_EPOCH=${LOAD_EPOCH:-50}

DATASET_CFG=${DATASET_CFG:-$COOP_ROOT/configs/datasets/${DATASET}.yaml}
FEATURE_JSON=${FEATURE_JSON:-$ROOT/outputs/task_features/${DATASET}_train.json}

COMMON="${DATASET}/shots_${SHOTS}/CoOpPriorRes/${TRAIN_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}_safe_noalt/seed${SEED}"
TRAIN_DIR="$COOP_ROOT/output/base2new/train_base/${COMMON}"

export PYTHONPATH="$SRC_ROOT:$COOP_ROOT:${PYTHONPATH:-}"
export USE_LEGACY_RESIDUAL=False

cd "$COOP_ROOT"

echo "============================================================"
echo "[B2N TRAIN BASE] dataset=${DATASET} seed=${SEED}"
echo "[OUT] ${TRAIN_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoOpPriorRes \
  --dataset-config-file "$DATASET_CFG" \
  --config-file "$TRAIN_CFG" \
  --output-dir "$TRAIN_DIR" \
  --seed "$SEED" \
  TRAINER.COOP.N_CTX "$NCTX" \
  TRAINER.COOP.CSC "$CSC" \
  TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
  DATASET.NUM_SHOTS "$SHOTS" \
  DATASET.SUBSAMPLE_CLASSES base \
  TRAINER.COOP.TASK_FEAT_PATH "$FEATURE_JSON" \
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

for SUB in base new; do
  TEST_DIR="$COOP_ROOT/output/base2new/test_${SUB}/${COMMON}"

  echo "============================================================"
  echo "[B2N TEST ${SUB}] dataset=${DATASET} seed=${SEED}"
  echo "[MODEL] ${TRAIN_DIR}"
  echo "[OUT] ${TEST_DIR}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "$DATA_ROOT" \
    --trainer CoOpPriorRes \
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
    DATASET.SUBSAMPLE_CLASSES "$SUB" \
    TRAINER.COOP.TASK_FEAT_PATH "$FEATURE_JSON" \
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
done

echo "[DONE] B2N PriorRes safe noalt dataset=${DATASET} seed=${SEED}"
