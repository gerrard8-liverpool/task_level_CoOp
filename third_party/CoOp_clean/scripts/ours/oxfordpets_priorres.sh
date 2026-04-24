#!/usr/bin/env bash
set -euo pipefail

cd /workspace/meta_prompt_1/third_party/CoOp
export PYTHONPATH=/workspace/meta_prompt_1/src:/workspace/meta_prompt_1/third_party/CoOp:${PYTHONPATH:-}

DATA_ROOT=/workspace/datasets
DATASET_CFG=configs/datasets/oxford_pets.yaml
TRAIN_CFG=configs/trainers/CoOp/rn50_ep50.yaml
TASK_FEAT=/workspace/meta_prompt_1/outputs/task_features/oxford_pets_train.json

SHOTS=${1:-4}
NCTX=${2:-16}
SEED=${3:-1}

OUT_DIR=output/oxford_pets/CoOpPriorRes/rn50_ep50_${SHOTS}shots/nctx${NCTX}_cscFalse_ctpend_seed${SEED}

python train.py \
  --root "${DATA_ROOT}" \
  --trainer CoOpPriorRes \
  --dataset-config-file "${DATASET_CFG}" \
  --config-file "${TRAIN_CFG}" \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  TRAINER.COOP.N_CTX "${NCTX}" \
  TRAINER.COOP.CSC False \
  TRAINER.COOP.CLASS_TOKEN_POSITION end \
  DATASET.NUM_SHOTS "${SHOTS}" \
  TRAINER.COOP.TASK_FEAT_PATH "${TASK_FEAT}" \
  TRAINER.COOP.TASK_FEAT_MODE transformed \
  TRAINER.COOP.META_HIDDEN_DIM 64 \
  TRAINER.COOP.META_KMAX 16 \
  TRAINER.COOP.GATE_TEMPERATURE 1.0 \
  TRAINER.COOP.INIT_GATE_BIAS 4.0 \
  TRAINER.COOP.USE_CONTEXT_GATING True \
  TRAINER.COOP.USE_B False \
  TRAINER.COOP.WARMUP_EPOCHS 5 \
  TRAINER.COOP.RAMP_EPOCHS 10 \
  TRAINER.COOP.LAMBDA_MAX 1.0 \
  TRAINER.COOP.ALTERNATE_OPT True \
  TRAINER.COOP.META_LR_RATIO 0.1 \
  TRAINER.COOP.DELTA_A_REG 1e-4 \
  TRAINER.COOP.DELTA_B_REG 1e-4
