#!/usr/bin/env bash
set -euo pipefail

cd /workspace/meta_prompt_1/third_party/CoOp_clean
export PYTHONPATH=/workspace/meta_prompt_1/src:/workspace/meta_prompt_1/third_party/CoOp_clean:/workspace/meta_prompt_1/third_party/Dassl.pytorch:${PYTHONPATH:-}

MODE=${1:-ab}      # aonly / bonly / ab
SEED=${2:-1}
GPU=${GPU:-0}
ROOT=${ROOT:-/workspace/datasets}

case "$MODE" in
  aonly)
    USE_CONTEXT_GATING=True
    USE_B=False
    ;;
  bonly)
    USE_CONTEXT_GATING=False
    USE_B=True
    ;;
  ab)
    USE_CONTEXT_GATING=True
    USE_B=True
    ;;
  *)
    echo "Unknown MODE=$MODE, expected one of: aonly / bonly / ab"
    exit 1
    ;;
esac

OUT_DIR=output/oxford_pets/CoOpPriorRes_stage2/${MODE}/rn50_16shots_beta0.2_seed${SEED}

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --root "${ROOT}" \
  --trainer CoOpPriorRes \
  --dataset-config-file configs/datasets/oxford_pets.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50.yaml \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  TRAINER.COOP.N_CTX 16 \
  TRAINER.COOP.CSC False \
  TRAINER.COOP.CLASS_TOKEN_POSITION end \
  DATASET.NUM_SHOTS 16 \
  TRAINER.COOP.TASK_FEAT_PATH /workspace/meta_prompt_1/outputs/task_features/oxford_pets_train.json \
  TRAINER.COOP.TASK_FEAT_MODE transformed \
  TRAINER.COOP.META_HIDDEN_DIM 64 \
  TRAINER.COOP.META_KMAX 16 \
  TRAINER.COOP.GATE_TEMPERATURE 1.0 \
  TRAINER.COOP.INIT_GATE_BIAS 1.0 \
  TRAINER.COOP.USE_CONTEXT_GATING ${USE_CONTEXT_GATING} \
  TRAINER.COOP.USE_B ${USE_B} \
  TRAINER.COOP.B_LOSS_WEIGHT 0.2 \
  TRAINER.COOP.WARMUP_EPOCHS 5 \
  TRAINER.COOP.RAMP_EPOCHS 10 \
  TRAINER.COOP.LAMBDA_MAX 1.0 \
  TRAINER.COOP.ALTERNATE_OPT True \
  TRAINER.COOP.META_LR_RATIO 0.3 \
  TRAINER.COOP.DELTA_A_REG 1e-4 \
  TRAINER.COOP.DELTA_B_REG 1e-4
