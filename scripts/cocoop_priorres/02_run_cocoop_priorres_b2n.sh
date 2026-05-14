#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
SRC_ROOT=${SRC_ROOT:-$ROOT/src}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

DATASET=${1:?Usage: GPU=0 bash 02_run_cocoop_priorres_b2n.sh <dataset> <seed>}
SEED=${2:?Usage: GPU=0 bash 02_run_cocoop_priorres_b2n.sh <dataset> <seed>}

GPU=${GPU:-0}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-4}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoCoOp/rn50_c4_ep10_batch1.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_c4_ep10_batch1}
LOAD_EPOCH=${LOAD_EPOCH:-10}

FEATURE_JSON=${FEATURE_JSON:-$ROOT/outputs/task_features_cocoop_b2n_base/rn50/${DATASET}_seed${SEED}_base_train.json}
DATASET_CFG="$COOP_ROOT/configs/datasets/${DATASET}.yaml"

if [ ! -f "$FEATURE_JSON" ]; then
  echo "[INFO] Missing B2N base-only feature, extracting now: $FEATURE_JSON"
  GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/00_extract_b2n_base_features.sh" "$DATASET" "$SEED"
fi

COMMON="${DATASET}/shots_${SHOTS}/CoCoOpPriorRes/${TRAIN_CFG_TAG}/nctx${NCTX}_ctxinit_a_photo_of_a_safe_noalt_baseprior/seed${SEED}"
TRAIN_DIR="$COOP_ROOT/output_cocoop_priorres/base2new/train_base/${COMMON}"

export PYTHONPATH="$SRC_ROOT:$COOP_ROOT:${PYTHONPATH:-}"
export USE_LEGACY_RESIDUAL=False

cd "$COOP_ROOT"

echo "============================================================"
echo "[CoCoOpPriorRes B2N TRAIN BASE]"
echo "dataset = $DATASET"
echo "seed    = $SEED"
echo "feature = $FEATURE_JSON"
echo "out     = $TRAIN_DIR"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoCoOpPriorRes \
  --dataset-config-file "$DATASET_CFG" \
  --config-file "$TRAIN_CFG" \
  --output-dir "$TRAIN_DIR" \
  --seed "$SEED" \
  DATASET.NUM_SHOTS "$SHOTS" \
  DATASET.SUBSAMPLE_CLASSES base \
  TRAINER.COCOOP.N_CTX "$NCTX" \
  TRAINER.COCOOP.CTX_INIT a_photo_of_a \
  TRAINER.COOP.TASK_FEAT_PATH "$FEATURE_JSON" \
  TRAINER.COOP.TASK_FEAT_MODE transformed \
  TRAINER.COOP.META_HIDDEN_DIM 64 \
  TRAINER.COOP.META_KMAX 16 \
  TRAINER.COOP.GATE_TEMPERATURE 1.0 \
  TRAINER.COOP.INIT_GATE_BIAS 4.0 \
  TRAINER.COOP.USE_CONTEXT_GATING True \
  TRAINER.COOP.USE_B False \
  TRAINER.COOP.WARMUP_EPOCHS 1 \
  TRAINER.COOP.RAMP_EPOCHS 3 \
  TRAINER.COOP.LAMBDA_MAX 0.5 \
  TRAINER.COOP.META_LR_RATIO 0.3 \
  TRAINER.COOP.DELTA_A_REG 1e-4 \
  TRAINER.COOP.DELTA_B_REG 1e-4

for SUB in base new; do
  TEST_DIR="$COOP_ROOT/output_cocoop_priorres/base2new/test_${SUB}/${COMMON}"

  echo "============================================================"
  echo "[CoCoOpPriorRes B2N TEST $SUB]"
  echo "dataset = $DATASET"
  echo "seed    = $SEED"
  echo "feature = $FEATURE_JSON"
  echo "model   = $TRAIN_DIR"
  echo "out     = $TEST_DIR"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "$DATA_ROOT" \
    --trainer CoCoOpPriorRes \
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
    TRAINER.COCOOP.CTX_INIT a_photo_of_a \
    TRAINER.COOP.TASK_FEAT_PATH "$FEATURE_JSON" \
    TRAINER.COOP.TASK_FEAT_MODE transformed \
    TRAINER.COOP.META_HIDDEN_DIM 64 \
    TRAINER.COOP.META_KMAX 16 \
    TRAINER.COOP.GATE_TEMPERATURE 1.0 \
    TRAINER.COOP.INIT_GATE_BIAS 4.0 \
    TRAINER.COOP.USE_CONTEXT_GATING True \
    TRAINER.COOP.USE_B False \
    TRAINER.COOP.WARMUP_EPOCHS 1 \
    TRAINER.COOP.RAMP_EPOCHS 3 \
    TRAINER.COOP.LAMBDA_MAX 0.5 \
    TRAINER.COOP.META_LR_RATIO 0.3 \
    TRAINER.COOP.DELTA_A_REG 1e-4 \
    TRAINER.COOP.DELTA_B_REG 1e-4
done
