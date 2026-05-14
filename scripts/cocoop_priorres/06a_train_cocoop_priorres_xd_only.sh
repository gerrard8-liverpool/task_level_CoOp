#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
SRC_ROOT=${SRC_ROOT:-$ROOT/src}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

SOURCE=${1:?Usage: GPU=0 bash 06a_train_cocoop_priorres_xd_only.sh <source> <seed>}
SEED=${2:?Usage: GPU=0 bash 06a_train_cocoop_priorres_xd_only.sh <source> <seed>}

GPU=${GPU:-0}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-4}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoCoOp/rn50_c4_ep10_batch1.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_c4_ep10_batch1}

SOURCE_FEATURE_JSON=${SOURCE_FEATURE_JSON:-$ROOT/outputs/task_features_cocoop_xd/rn50/${SOURCE}_seed${SEED}_train.json}
SOURCE_CFG="$COOP_ROOT/configs/datasets/${SOURCE}.yaml"

if [ ! -f "$SOURCE_FEATURE_JSON" ]; then
  GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/00_extract_dg_source_features_if_missing.sh" "$SOURCE" "$SEED"
fi

COMMON="source_${SOURCE}/shots_${SHOTS}/CoCoOpPriorRes/${TRAIN_CFG_TAG}/nctx${NCTX}_ctxinit_a_photo_of_a_safe_noalt_sourceprior/seed${SEED}"
TRAIN_DIR="$COOP_ROOT/output_cocoop_priorres/xd/train/${COMMON}"

if [ -f "$TRAIN_DIR/prompt_learner/model.pth.tar-10" ] && [ -f "$TRAIN_DIR/prior_adapter/model.pth.tar-10" ]; then
  echo "[SKIP] CoCoOpPriorRes train done: $TRAIN_DIR"
  exit 0
fi

export PYTHONPATH="$SRC_ROOT:$COOP_ROOT:${PYTHONPATH:-}"
export USE_LEGACY_RESIDUAL=False

cd "$COOP_ROOT"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoCoOpPriorRes \
  --dataset-config-file "$SOURCE_CFG" \
  --config-file "$TRAIN_CFG" \
  --output-dir "$TRAIN_DIR" \
  --seed "$SEED" \
  DATASET.NUM_SHOTS "$SHOTS" \
  DATASET.SUBSAMPLE_CLASSES all \
  TRAINER.COCOOP.N_CTX "$NCTX" \
  TRAINER.COCOOP.CTX_INIT a_photo_of_a \
  TRAINER.COOP.TASK_FEAT_PATH "$SOURCE_FEATURE_JSON" \
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
