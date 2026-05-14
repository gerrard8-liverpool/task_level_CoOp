#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}

GPU=${GPU:-0}
DATASETS=(caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101)
SEEDS=(1 2 3)

COCOOP_SUBDIR="CoCoOp/rn50_c4_ep10_batch1/nctx4_ctxinit_a_photo_of_a"
PRIOR_SUBDIR="CoCoOpPriorRes/rn50_c4_ep10_batch1/nctx4_ctxinit_a_photo_of_a_safe_noalt_baseprior"

is_done_cocoop() {
  local D=$1
  local S=$2
  local base_log="$COOP_ROOT/output_cocoop_priorres/base2new/test_base/${D}/shots_16/${COCOOP_SUBDIR}/seed${S}/log.txt"
  local new_log="$COOP_ROOT/output_cocoop_priorres/base2new/test_new/${D}/shots_16/${COCOOP_SUBDIR}/seed${S}/log.txt"
  [[ -f "$base_log" && -f "$new_log" ]] && grep -q "accuracy" "$base_log" && grep -q "accuracy" "$new_log"
}

is_done_prior() {
  local D=$1
  local S=$2
  local base_log="$COOP_ROOT/output_cocoop_priorres/base2new/test_base/${D}/shots_16/${PRIOR_SUBDIR}/seed${S}/log.txt"
  local new_log="$COOP_ROOT/output_cocoop_priorres/base2new/test_new/${D}/shots_16/${PRIOR_SUBDIR}/seed${S}/log.txt"
  [[ -f "$base_log" && -f "$new_log" ]] && grep -q "accuracy" "$base_log" && grep -q "accuracy" "$new_log"
}

echo "============================================================"
echo "[B2N full skip-done]"
echo "GPU      = $GPU"
echo "datasets = ${DATASETS[*]}"
echo "seeds    = ${SEEDS[*]}"
echo "============================================================"

for D in "${DATASETS[@]}"; do
  for S in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "[B2N] dataset=$D seed=$S gpu=$GPU"
    echo "============================================================"

    GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/00_extract_b2n_base_features.sh" "$D" "$S"

    if is_done_cocoop "$D" "$S"; then
      echo "[SKIP] CoCoOp B2N already done: dataset=$D seed=$S"
    else
      GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/01_run_cocoop_b2n.sh" "$D" "$S"
    fi

    if is_done_prior "$D" "$S"; then
      echo "[SKIP] CoCoOpPriorRes B2N already done: dataset=$D seed=$S"
    else
      GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/02_run_cocoop_priorres_b2n.sh" "$D" "$S"
    fi

    python "$ROOT/scripts/cocoop_priorres/09_summarize_cocoop_b2n.py"
  done
done

python "$ROOT/scripts/cocoop_priorres/09_summarize_cocoop_b2n.py"
cat "$ROOT/outputs/cocoop_priorres_b2n_summary.md"
