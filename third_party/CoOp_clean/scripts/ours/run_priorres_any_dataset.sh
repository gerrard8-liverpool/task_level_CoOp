#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
SRC_ROOT=${SRC_ROOT:-$ROOT/src}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}

DATASET=${1:?Usage: bash run_priorres_any_dataset.sh <dataset> [shots] [nctx] [seed]}
SHOTS=${2:-4}
NCTX=${3:-16}
SEED=${4:-1}

CTX_POS=${CTX_POS:-end}
CSC=${CSC:-False}
TRAINER_NAME=${TRAINER_NAME:-CoOpPriorRes}
TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoOp/rn50_ep50.yaml}
BACKBONE=${BACKBONE:-RN50}
FEATURE_MODE=${FEATURE_MODE:-transformed}
FEATURE_SPLIT=${FEATURE_SPLIT:-train_x}
FEATURE_JSON=${FEATURE_JSON:-$ROOT/outputs/task_features/${DATASET}_train.json}
FORCE_EXTRACT=${FORCE_EXTRACT:-0}

USE_CONTEXT_GATING=${USE_CONTEXT_GATING:-True}
USE_LEGACY_RESIDUAL=${USE_LEGACY_RESIDUAL:-False}
USE_B=${USE_B:-False}
META_HIDDEN_DIM=${META_HIDDEN_DIM:-64}
META_KMAX=${META_KMAX:-16}
GATE_TEMPERATURE=${GATE_TEMPERATURE:-1.0}
INIT_GATE_BIAS=${INIT_GATE_BIAS:-4.0}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
RAMP_EPOCHS=${RAMP_EPOCHS:-10}
LAMBDA_MAX=${LAMBDA_MAX:-1.0}
ALTERNATE_OPT=${ALTERNATE_OPT:-True}
META_LR_RATIO=${META_LR_RATIO:-0.1}
DELTA_A_REG=${DELTA_A_REG:-1e-4}
DELTA_B_REG=${DELTA_B_REG:-1e-4}
B_LOSS_WEIGHT=${B_LOSS_WEIGHT:-1.0}

DATASET_CFG=${DATASET_CFG:-$COOP_ROOT/configs/datasets/${DATASET}.yaml}
TRAIN_CFG_TAG=$(basename "$TRAIN_CFG" .yaml)
OUT_DIR=${OUT_DIR:-$COOP_ROOT/output/${DATASET}/${TRAINER_NAME}/${TRAIN_CFG_TAG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}_seed${SEED}}

EXTRA_TRAIN_OPTS=("$@")
EXTRA_TRAIN_OPTS=("${EXTRA_TRAIN_OPTS[@]:4}")

if [[ ! -d "$COOP_ROOT" ]]; then
  echo "[ERROR] COOP_ROOT not found: $COOP_ROOT" >&2
  exit 1
fi

if [[ ! -f "$COOP_ROOT/train.py" ]]; then
  echo "[ERROR] train.py not found under $COOP_ROOT" >&2
  exit 1
fi

if [[ ! -f "$DATASET_CFG" ]]; then
  echo "[ERROR] dataset config not found: $DATASET_CFG" >&2
  exit 1
fi

if [[ ! -f "$COOP_ROOT/$TRAIN_CFG" ]]; then
  echo "[ERROR] train config not found: $COOP_ROOT/$TRAIN_CFG" >&2
  exit 1
fi

export PYTHONPATH="$SRC_ROOT:$COOP_ROOT:${PYTHONPATH:-}"
export USE_LEGACY_RESIDUAL
mkdir -p "$(dirname "$FEATURE_JSON")"
mkdir -p "$OUT_DIR"

BOOL_TRUE_SET=(True true 1 YES yes Y y)
contains_true() {
  local x="$1"
  for t in "${BOOL_TRUE_SET[@]}"; do
    if [[ "$x" == "$t" ]]; then
      return 0
    fi
  done
  return 1
}

maybe_extract_feature() {
  if [[ -f "$FEATURE_JSON" && "$FORCE_EXTRACT" != "1" ]]; then
    echo "[INFO] Reusing existing task feature: $FEATURE_JSON"
    return 0
  fi

  echo "[INFO] Extracting task feature -> $FEATURE_JSON"
  python "$SRC_ROOT/meta_prompts/task_feature_extractor.py" \
    --coop-root "$COOP_ROOT" \
    --root "$DATA_ROOT" \
    --dataset-config-file "$DATASET_CFG" \
    --backbone "$BACKBONE" \
    --split "$FEATURE_SPLIT" \
    --text-template '{}' \
    --output "$FEATURE_JSON"
}

check_b_branch_constraints() {
  if ! contains_true "$USE_B"; then
    return 0
  fi

  local slotproto="$DATA_ROOT/${DATASET}/split_fewshot/shot_${SHOTS}-seed_${SEED}-slotproto.pkl"

  if [[ ! -f "$slotproto" ]]; then
    echo "[INFO] Missing slotproto cache: $slotproto"
    echo "[INFO] Building slotproto cache for dataset=$DATASET shots=$SHOTS seed=$SEED"

    python "$ROOT/scripts/ours/build_slotproto_cache_any_dataset.py" \
      --dataset-root "$DATA_ROOT" \
      --dataset-name "$DATASET" \
      --backbone "$BACKBONE" \
      --seeds "$SEED" \
      --kmax "$SHOTS"
  fi

  if [[ ! -f "$slotproto" ]]; then
    echo "[ERROR] slotproto cache still missing after build: $slotproto" >&2
    exit 1
  fi
}

run_train() {
  echo "[INFO] Launching training"
  echo "       dataset      = $DATASET"
  echo "       shots (k)    = $SHOTS"
  echo "       nctx  (m)    = $NCTX"
  echo "       seed         = $SEED"
  echo "       use_b        = $USE_B"
  echo "       legacy_res   = $USE_LEGACY_RESIDUAL"
  echo "       feature_json = $FEATURE_JSON"
  echo "       output_dir   = $OUT_DIR"

  cd "$COOP_ROOT"
  python train.py \
    --root "$DATA_ROOT" \
    --trainer "$TRAINER_NAME" \
    --dataset-config-file "$DATASET_CFG" \
    --config-file "$TRAIN_CFG" \
    --output-dir "$OUT_DIR" \
    --seed "$SEED" \
    TRAINER.COOP.N_CTX "$NCTX" \
    TRAINER.COOP.CSC "$CSC" \
    TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
    DATASET.NUM_SHOTS "$SHOTS" \
    TRAINER.COOP.TASK_FEAT_PATH "$FEATURE_JSON" \
    TRAINER.COOP.TASK_FEAT_MODE "$FEATURE_MODE" \
    TRAINER.COOP.META_HIDDEN_DIM "$META_HIDDEN_DIM" \
    TRAINER.COOP.META_KMAX "$META_KMAX" \
    TRAINER.COOP.GATE_TEMPERATURE "$GATE_TEMPERATURE" \
    TRAINER.COOP.INIT_GATE_BIAS "$INIT_GATE_BIAS" \
    TRAINER.COOP.USE_CONTEXT_GATING "$USE_CONTEXT_GATING" \
    TRAINER.COOP.USE_B "$USE_B" \
    TRAINER.COOP.B_LOSS_WEIGHT "$B_LOSS_WEIGHT" \
    TRAINER.COOP.WARMUP_EPOCHS "$WARMUP_EPOCHS" \
    TRAINER.COOP.RAMP_EPOCHS "$RAMP_EPOCHS" \
    TRAINER.COOP.LAMBDA_MAX "$LAMBDA_MAX" \
    TRAINER.COOP.ALTERNATE_OPT "$ALTERNATE_OPT" \
    TRAINER.COOP.META_LR_RATIO "$META_LR_RATIO" \
    TRAINER.COOP.DELTA_A_REG "$DELTA_A_REG" \
    TRAINER.COOP.DELTA_B_REG "$DELTA_B_REG" \
    "${EXTRA_TRAIN_OPTS[@]}"
}

check_b_branch_constraints
maybe_extract_feature
run_train
