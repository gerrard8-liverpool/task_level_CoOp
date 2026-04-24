#!/usr/bin/env bash
set -euo pipefail

cd /workspace/meta_prompt_1/third_party/CoOp_clean
export PYTHONPATH=/workspace/meta_prompt_1/src:/workspace/meta_prompt_1/third_party/CoOp_clean:$PYTHONPATH

ROOT=${ROOT:-/workspace/datasets}
GPU=${GPU:-0}
MODE=${MODE:-both}   # both / coop / priorres
SEEDS=${SEEDS:-"1 2 3"}
K_VALUES=${K_VALUES:-"1 2 4 8 10 12 14 16"}

COMMON_CFG=(
  --root "$ROOT"
  --dataset-config-file configs/datasets/oxford_pets.yaml
  --config-file configs/trainers/CoOp/rn50_ep50.yaml
)

run_coop () {
  local K=$1
  local SEED=$2
  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    "${COMMON_CFG[@]}" \
    --trainer CoOp \
    --output-dir output/oxford_pets/CoOp/rn50_${K}shots_seed${SEED} \
    --seed ${SEED} \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC False \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS ${K}
}

run_priorres () {
  local K=$1
  local SEED=$2
  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    "${COMMON_CFG[@]}" \
    --trainer CoOpPriorRes \
    --output-dir output/oxford_pets/CoOpPriorRes_datasetonly/rn50_${K}shots_b1.0_lr0.3_seed${SEED} \
    --seed ${SEED} \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC False \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS ${K} \
    TRAINER.COOP.TASK_FEAT_PATH /workspace/meta_prompt_1/outputs/task_features/oxford_pets_train.json \
    TRAINER.COOP.TASK_FEAT_MODE transformed \
    TRAINER.COOP.META_HIDDEN_DIM 64 \
    TRAINER.COOP.META_KMAX 16 \
    TRAINER.COOP.GATE_TEMPERATURE 1.0 \
    TRAINER.COOP.INIT_GATE_BIAS 1.0 \
    TRAINER.COOP.USE_CONTEXT_GATING True \
    TRAINER.COOP.USE_B False \
    TRAINER.COOP.WARMUP_EPOCHS 5 \
    TRAINER.COOP.RAMP_EPOCHS 10 \
    TRAINER.COOP.LAMBDA_MAX 1.0 \
    TRAINER.COOP.ALTERNATE_OPT True \
    TRAINER.COOP.META_LR_RATIO 0.3 \
    TRAINER.COOP.DELTA_A_REG 1e-4 \
    TRAINER.COOP.DELTA_B_REG 1e-4
}

echo "MODE=$MODE"
echo "GPU=$GPU"
echo "SEEDS=$SEEDS"
echo "K_VALUES=$K_VALUES"

for K in ${K_VALUES}; do
  for SEED in ${SEEDS}; do
    echo "=============================================================="
    echo "K=${K} | SEED=${SEED}"
    echo "=============================================================="
    if [[ "$MODE" == "both" || "$MODE" == "coop" ]]; then
      echo "[RUN] CoOp | K=${K} | SEED=${SEED}"
      run_coop ${K} ${SEED}
    fi
    if [[ "$MODE" == "both" || "$MODE" == "priorres" ]]; then
      echo "[RUN] CoOpPriorRes_datasetonly | K=${K} | SEED=${SEED}"
      run_priorres ${K} ${SEED}
    fi
  done
done

echo "All runs finished."
