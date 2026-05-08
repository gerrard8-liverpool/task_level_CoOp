#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPU=0 bash scripts/ablations/safe_prior/run_safe_prior_xd.sh <tag> <source> <task_feat_path> <seed> <target1> [target2 ...]

TAG=${1:?tag required, e.g. safe_mean or safe_shuffle_foodfeat}
SOURCE=${2:?source dataset required}
TASK_FEAT_PATH=${3:?task feature path required}
SEED=${4:?seed required}
shift 4

if [ "$#" -lt 1 ]; then
  echo "At least one target dataset is required."
  exit 1
fi

TARGETS=("$@")

GPU=${GPU:-0}
ROOT="/workspace/datasets"
PROJECT_ROOT="/workspace/meta_prompt_1"
COOP_ROOT="${PROJECT_ROOT}/third_party/CoOp_clean"

TRAINER="CoOpPriorRes"
TRAIN_CFG="configs/trainers/CoOp/rn50_ep50.yaml"
TRAIN_CFG_TAG="rn50_ep50"

NCTX=16
SHOTS=16
CSC=False
CTX_POS=end

TASK_FEAT_PATH_ABS=$(readlink -f "$TASK_FEAT_PATH")

COMMON="source_${SOURCE}/shots_${SHOTS}/${TRAINER}/${TRAIN_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}_${TAG}/seed${SEED}"
TRAIN_DIR="${PROJECT_ROOT}/outputs/ablations/safe_prior/runs/xd/train/${COMMON}"

echo "============================================================"
echo "[Safe Prior Ablation TRAIN]"
echo "TAG=${TAG}"
echo "SOURCE=${SOURCE}"
echo "SEED=${SEED}"
echo "TASK_FEAT_PATH=${TASK_FEAT_PATH_ABS}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TARGETS=${TARGETS[*]}"
echo "GPU=${GPU}"
echo "============================================================"

if [ ! -f "$TASK_FEAT_PATH_ABS" ]; then
  echo "[ERROR] Missing task feature: $TASK_FEAT_PATH_ABS"
  exit 1
fi

cd "$COOP_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export USE_CONTEXT_GATING=True
export USE_LEGACY_RESIDUAL=False
export USE_B=False
export ALTERNATE_OPT=False
export META_LR_RATIO=0.3

if [ -f "${TRAIN_DIR}/prompt_learner/model.pth.tar-50" ] && [ -f "${TRAIN_DIR}/prior_adapter/model.pth.tar-50" ]; then
  echo "[SKIP TRAIN] Found checkpoint: ${TRAIN_DIR}"
else
  python train.py \
    --root "$ROOT" \
    --seed "$SEED" \
    --trainer "$TRAINER" \
    --dataset-config-file "configs/datasets/${SOURCE}.yaml" \
    --config-file "$TRAIN_CFG" \
    --output-dir "$TRAIN_DIR" \
    TRAINER.COOP.N_CTX "$NCTX" \
    TRAINER.COOP.CSC "$CSC" \
    TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
    DATASET.NUM_SHOTS "$SHOTS" \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.COOP.TASK_FEAT_PATH "$TASK_FEAT_PATH_ABS" \
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

for TARGET in "${TARGETS[@]}"; do
  TEST_DIR="${PROJECT_ROOT}/outputs/ablations/safe_prior/runs/xd/test/${TARGET}/${COMMON}"

  if [ -f "${TEST_DIR}/log.txt" ] && grep -q "accuracy:" "${TEST_DIR}/log.txt"; then
    echo "[SKIP TEST] ${SOURCE} -> ${TARGET}, seed=${SEED}, tag=${TAG}"
    continue
  fi

  echo "[Safe Prior Ablation TEST] ${SOURCE} -> ${TARGET}, seed=${SEED}, tag=${TAG}"

  python train.py \
    --root "$ROOT" \
    --seed "$SEED" \
    --trainer "$TRAINER" \
    --dataset-config-file "configs/datasets/${TARGET}.yaml" \
    --config-file "$TRAIN_CFG" \
    --eval-only \
    --model-dir "$TRAIN_DIR" \
    --load-epoch 50 \
    --output-dir "$TEST_DIR" \
    TRAINER.COOP.N_CTX "$NCTX" \
    TRAINER.COOP.CSC "$CSC" \
    TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
    DATASET.NUM_SHOTS "$SHOTS" \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.COOP.TASK_FEAT_PATH "$TASK_FEAT_PATH_ABS" \
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

echo "[DONE] Safe prior ablation tag=${TAG}, source=${SOURCE}, seed=${SEED}"
