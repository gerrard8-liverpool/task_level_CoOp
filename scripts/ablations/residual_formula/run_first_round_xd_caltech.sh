#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
cd "$ROOT"

GPU=${GPU:-0}
SOURCE=${SOURCE:-caltech101}
SEEDS=(${SEEDS:-1 2 3})
ALL_DATASETS=(${ALL_DATASETS:-oxford_pets eurosat dtd food101 oxford_flowers stanford_cars fgvc_aircraft ucf101 sun397})
TARGETS=(${TARGETS:-${ALL_DATASETS[*]}})

function has_accuracy() {
  local log="$1"
  if [ -f "$log" ] && grep -q "accuracy:" "$log"; then
    return 0
  fi
  return 1
}

function coop_target_log() {
  local target="$1"
  local seed="$2"
  echo "$ROOT/outputs/ablations/residual_formula/runs/xd/test/${target}/source_${SOURCE}/shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed${seed}/log.txt"
}

function prior_target_log() {
  local target="$1"
  local variant="$2"
  local seed="$3"
  echo "$ROOT/outputs/ablations/residual_formula/runs/xd/test/${target}/source_${SOURCE}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_${variant}/seed${seed}/log.txt"
}

function all_targets_done_coop() {
  local seed="$1"
  for target in "${TARGETS[@]}"; do
    if ! has_accuracy "$(coop_target_log "$target" "$seed")"; then
      return 1
    fi
  done
  return 0
}

function all_targets_done_prior() {
  local variant="$1"
  local seed="$2"
  for target in "${TARGETS[@]}"; do
    if ! has_accuracy "$(prior_target_log "$target" "$variant" "$seed")"; then
      return 1
    fi
  done
  return 0
}

echo "============================================================"
echo "[FIRST ROUND XD RESIDUAL FORMULA ABLATION]"
echo "source  = ${SOURCE}"
echo "targets = ${TARGETS[*]}"
echo "seeds   = ${SEEDS[*]}"
echo "gpu     = ${GPU}"
echo "============================================================"

mkdir -p outputs/ablations/residual_formula/{runs,tables,logs}

for seed in "${SEEDS[@]}"; do
  if all_targets_done_coop "$seed"; then
    echo "[SKIP] XD CoOp done: source=${SOURCE}, seed=${seed}"
  else
    echo "[RUN] XD CoOp: source=${SOURCE}, seed=${seed}"
    GPU=${GPU} bash scripts/ablations/residual_formula/run_coop_residual_xd.sh "$SOURCE" "$seed" "${TARGETS[@]}"
  fi

  for variant in legacy safe; do
    if all_targets_done_prior "$variant" "$seed"; then
      echo "[SKIP] XD PriorRes ${variant} done: source=${SOURCE}, seed=${seed}"
    else
      echo "[RUN] XD PriorRes ${variant}: source=${SOURCE}, seed=${seed}"
      GPU=${GPU} TEST_FEATURE_MODE=source bash scripts/ablations/residual_formula/run_priorres_residual_xd.sh "$variant" "$SOURCE" "$seed" "${TARGETS[@]}"
    fi
  done

done

python scripts/ablations/residual_formula/summarize_residual_formula.py \
  --mode xd \
  --source "$SOURCE" \
  --targets "${TARGETS[@]}" \
  > outputs/ablations/residual_formula/tables/xd_${SOURCE}_residual_formula.md

cat outputs/ablations/residual_formula/tables/xd_${SOURCE}_residual_formula.md
