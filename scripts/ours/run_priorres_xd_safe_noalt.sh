#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
SRC_ROOT=${SRC_ROOT:-$ROOT/src}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

SOURCE=${1:?Usage: bash run_priorres_xd_safe_noalt.sh <source_dataset> <seed> <target1> [target2 ...]}
SEED=${2:?Usage: bash run_priorres_xd_safe_noalt.sh <source_dataset> <seed> <target1> [target2 ...]}
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
SOURCE_FEATURE_JSON=${SOURCE_FEATURE_JSON:-$ROOT/outputs/task_features/${SOURCE}_train.json}

# DG strict setting:
#   source = use source task feature during both source training and target evaluation
#   target = use target task feature during target evaluation; exploratory only, less strict
TEST_FEATURE_MODE=${TEST_FEATURE_MODE:-source}

COMMON="source_${SOURCE}/shots_${SHOTS}/CoOpPriorRes/${TRAIN_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}_safe_noalt/seed${SEED}"
TRAIN_DIR="$COOP_ROOT/output/xd/train/${COMMON}"

export PYTHONPATH="$SRC_ROOT:$COOP_ROOT:${PYTHONPATH:-}"
export USE_LEGACY_RESIDUAL=False

cd "$COOP_ROOT"

echo "============================================================"
echo "[XD PriorRes TRAIN SOURCE]"
echo "source      = ${SOURCE}"
echo "seed        = ${SEED}"
echo "out         = ${TRAIN_DIR}"
echo "feature     = ${SOURCE_FEATURE_JSON}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "$DATA_ROOT" \
  --trainer CoOpPriorRes \
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
  DATASET.SUBSAMPLE_CLASSES all \
  TRAINER.COOP.TASK_FEAT_PATH "$SOURCE_FEATURE_JSON" \
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

for TARGET in "${TARGETS[@]}"; do
  TARGET_CFG="$COOP_ROOT/configs/datasets/${TARGET}.yaml"

  if [ "$TEST_FEATURE_MODE" = "target" ]; then
    TEST_FEATURE_JSON="$ROOT/outputs/task_features/${TARGET}_train.json"
  else
    TEST_FEATURE_JSON="$SOURCE_FEATURE_JSON"
  fi

  TEST_DIR="$COOP_ROOT/output/xd/test/${TARGET}/${COMMON}"

  echo "============================================================"
  echo "[XD PriorRes TEST TARGET]"
  echo "source      = ${SOURCE}"
  echo "target      = ${TARGET}"
  echo "seed        = ${SEED}"
  echo "model       = ${TRAIN_DIR}"
  echo "out         = ${TEST_DIR}"
  echo "feat_mode   = ${TEST_FEATURE_MODE}"
  echo "feature     = ${TEST_FEATURE_JSON}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "$DATA_ROOT" \
    --trainer CoOpPriorRes \
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
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.COOP.TASK_FEAT_PATH "$TEST_FEATURE_JSON" \
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

echo "[DONE] XD PriorRes source=${SOURCE} seed=${SEED}"
