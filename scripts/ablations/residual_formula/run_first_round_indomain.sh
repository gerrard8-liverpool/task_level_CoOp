#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
cd "$ROOT"

GPU=${GPU:-0}
DATASETS=(${DATASETS:-eurosat dtd caltech101})
SEEDS=(${SEEDS:-1 2 3})

function has_accuracy() {
  local log="$1"
  if [ -f "$log" ] && grep -q "accuracy:" "$log"; then
    return 0
  fi
  return 1
}

function coop_log() {
  local dataset="$1"
  local seed="$2"
  echo "$ROOT/outputs/ablations/residual_formula/runs/indomain/${dataset}/coop/shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed${seed}/log.txt"
}

function prior_log() {
  local dataset="$1"
  local variant="$2"
  local seed="$3"
  echo "$ROOT/outputs/ablations/residual_formula/runs/indomain/${dataset}/${variant}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend/seed${seed}/log.txt"
}

echo "============================================================"
echo "[FIRST ROUND IN-DOMAIN RESIDUAL FORMULA ABLATION]"
echo "datasets = ${DATASETS[*]}"
echo "seeds    = ${SEEDS[*]}"
echo "gpu      = ${GPU}"
echo "============================================================"

mkdir -p outputs/ablations/residual_formula/{runs,tables,logs}

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if has_accuracy "$(coop_log "$dataset" "$seed")"; then
      echo "[SKIP] CoOp done: dataset=${dataset}, seed=${seed}"
    else
      echo "[RUN] CoOp: dataset=${dataset}, seed=${seed}"
      GPU=${GPU} bash scripts/ablations/residual_formula/run_coop_residual_indomain.sh "$dataset" "$seed"
    fi

    for variant in legacy safe; do
      if has_accuracy "$(prior_log "$dataset" "$variant" "$seed")"; then
        echo "[SKIP] PriorRes ${variant} done: dataset=${dataset}, seed=${seed}"
      else
        echo "[RUN] PriorRes ${variant}: dataset=${dataset}, seed=${seed}"
        GPU=${GPU} bash scripts/ablations/residual_formula/run_priorres_residual_indomain.sh "$variant" "$dataset" "$seed"
      fi
    done
  done

done

python scripts/ablations/residual_formula/summarize_residual_formula.py \
  --mode indomain \
  --datasets "${DATASETS[@]}" \
  > outputs/ablations/residual_formula/tables/indomain_residual_formula.md

cat outputs/ablations/residual_formula/tables/indomain_residual_formula.md
