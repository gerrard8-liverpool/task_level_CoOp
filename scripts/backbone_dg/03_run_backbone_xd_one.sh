#!/usr/bin/env bash
set -euo pipefail

# Generic backbone XD runner with skip logic.
# Usage:
#   GPU=0 bash scripts/backbone_dg/03_run_backbone_xd_one.sh coop rn101 caltech101 1 eurosat dtd sun397
#   GPU=0 bash scripts/backbone_dg/03_run_backbone_xd_one.sh safe rn101 caltech101 1 eurosat dtd sun397

METHOD=${1:?method required: coop|safe}
BACKBONE_TAG=${2:?backbone tag required: rn101|vit_b16|vit_b32}
SOURCE=${3:?source dataset required}
SEED=${4:?seed required}
shift 4
TARGETS=("$@")

if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "[ERROR] Please provide at least one target dataset."
  exit 1
fi

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
SRC_ROOT=${SRC_ROOT:-$ROOT/src}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CTX_POS=${CTX_POS:-end}
CSC=${CSC:-False}
LOAD_EPOCH=${LOAD_EPOCH:-50}

case "$BACKBONE_TAG" in
  rn101)
    CFG_TAG="rn101_ep50"
    ;;
  vit_b16)
    CFG_TAG="vit_b16_ep50"
    ;;
  vit_b32)
    CFG_TAG="vit_b32_ep50"
    ;;
  *)
    echo "[ERROR] Unknown BACKBONE_TAG=$BACKBONE_TAG"
    exit 1
    ;;
esac

TRAIN_CFG="configs/trainers/CoOp/${CFG_TAG}.yaml"
SOURCE_CFG="$COOP_ROOT/configs/datasets/${SOURCE}.yaml"

if [ ! -f "$COOP_ROOT/$TRAIN_CFG" ]; then
  echo "[ERROR] Missing train config: $COOP_ROOT/$TRAIN_CFG"
  echo "Run: bash scripts/backbone_dg/01_create_backbone_configs.sh"
  exit 1
fi

cd "$COOP_ROOT"

if [ "$METHOD" = "coop" ]; then
  TRAINER="CoOp"
  COMMON="source_${SOURCE}/shots_${SHOTS}/CoOp/${CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}"
  TRAIN_DIR="$COOP_ROOT/output/xd/train/${COMMON}"
  TRAIN_OK="$TRAIN_DIR/prompt_learner/model.pth.tar-${LOAD_EPOCH}"
elif [ "$METHOD" = "safe" ]; then
  TRAINER="CoOpPriorRes"
  export PYTHONPATH="$SRC_ROOT:$COOP_ROOT:${PYTHONPATH:-}"
  export USE_LEGACY_RESIDUAL=False
  SOURCE_FEATURE_JSON=${SOURCE_FEATURE_JSON:-$ROOT/outputs/task_features_backbone/${BACKBONE_TAG}/${SOURCE}_train.json}
  if [ ! -f "$SOURCE_FEATURE_JSON" ]; then
    echo "[ERROR] Missing source feature: $SOURCE_FEATURE_JSON"
    echo "Run: GPU=$GPU bash scripts/backbone_dg/02_extract_backbone_task_features.sh $BACKBONE_TAG $SOURCE"
    exit 1
  fi
  COMMON="source_${SOURCE}/shots_${SHOTS}/CoOpPriorRes/${CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}_safe_noalt/seed${SEED}"
  TRAIN_DIR="$COOP_ROOT/output/xd/train/${COMMON}"
  TRAIN_OK="$TRAIN_DIR/prompt_learner/model.pth.tar-${LOAD_EPOCH}"
else
  echo "[ERROR] Unknown METHOD=$METHOD. Use coop or safe."
  exit 1
fi

if [ -f "$TRAIN_OK" ]; then
  echo "[SKIP TRAIN] $METHOD $BACKBONE_TAG source=$SOURCE seed=$SEED"
else
  echo "============================================================"
  echo "[TRAIN] method=$METHOD backbone=$BACKBONE_TAG source=$SOURCE seed=$SEED"
  echo "out=$TRAIN_DIR"
  echo "============================================================"

  if [ "$METHOD" = "coop" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --root "$DATA_ROOT" \
      --trainer "$TRAINER" \
      --dataset-config-file "$SOURCE_CFG" \
      --config-file "$TRAIN_CFG" \
      --output-dir "$TRAIN_DIR" \
      --seed "$SEED" \
      TRAINER.COOP.N_CTX "$NCTX" \
      TRAINER.COOP.CSC "$CSC" \
      TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
      DATASET.NUM_SHOTS "$SHOTS" \
      DATASET.SUBSAMPLE_CLASSES all
  else
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --root "$DATA_ROOT" \
      --trainer "$TRAINER" \
      --dataset-config-file "$SOURCE_CFG" \
      --config-file "$TRAIN_CFG" \
      --output-dir "$TRAIN_DIR" \
      --seed "$SEED" \
      TRAINER.COOP.N_CTX "$NCTX" \
      TRAINER.COOP.CSC "$CSC" \
      TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
      DATASET.NUM_SHOTS "$SHOTS" \
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
  fi
fi

for TARGET in "${TARGETS[@]}"; do
  TARGET_CFG="$COOP_ROOT/configs/datasets/${TARGET}.yaml"
  TEST_DIR="$COOP_ROOT/output/xd/test/${TARGET}/${COMMON}"

  if [ -f "$TEST_DIR/log.txt" ] && grep -q "accuracy:" "$TEST_DIR/log.txt"; then
    echo "[SKIP TEST] method=$METHOD backbone=$BACKBONE_TAG $SOURCE->$TARGET seed=$SEED"
    continue
  fi

  echo "============================================================"
  echo "[TEST] method=$METHOD backbone=$BACKBONE_TAG $SOURCE->$TARGET seed=$SEED"
  echo "model=$TRAIN_DIR"
  echo "out=$TEST_DIR"
  echo "============================================================"

  if [ "$METHOD" = "coop" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --root "$DATA_ROOT" \
      --trainer "$TRAINER" \
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
      DATASET.NUM_SHOTS "$SHOTS" \
      DATASET.SUBSAMPLE_CLASSES all
  else
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --root "$DATA_ROOT" \
      --trainer "$TRAINER" \
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
      DATASET.NUM_SHOTS "$SHOTS" \
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
  fi
done

 echo "[DONE] method=$METHOD backbone=$BACKBONE_TAG source=$SOURCE seed=$SEED"
