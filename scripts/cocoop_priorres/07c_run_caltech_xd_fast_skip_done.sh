#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
GPU=${GPU:-0}
SOURCE=${SOURCE:-caltech101}
SEEDS=(1 2 3)

# fast subset: 先不跑 sun397；food101 也先不跑，避免大测试集压 I/O
TARGETS=(dtd eurosat fgvc_aircraft oxford_flowers oxford_pets stanford_cars ucf101)

COCOOP_SUBDIR="CoCoOp/rn50_c4_ep10_batch1/nctx4_ctxinit_a_photo_of_a"
PRIOR_SUBDIR="CoCoOpPriorRes/rn50_c4_ep10_batch1/nctx4_ctxinit_a_photo_of_a_safe_noalt_sourceprior"

has_acc() {
  local f="$1"
  [[ -f "$f" ]] && grep -q "accuracy" "$f"
}

for SEED in "${SEEDS[@]}"; do
  for TARGET in "${TARGETS[@]}"; do
    COCOOP_LOG="$COOP_ROOT/output_cocoop_priorres/xd/test/${TARGET}/source_${SOURCE}/shots_16/${COCOOP_SUBDIR}/seed${SEED}/log.txt"
    PRIOR_LOG="$COOP_ROOT/output_cocoop_priorres/xd/test/${TARGET}/source_${SOURCE}/shots_16/${PRIOR_SUBDIR}/seed${SEED}/log.txt"

    echo "============================================================"
    echo "[FAST XD] source=${SOURCE} target=${TARGET} seed=${SEED} gpu=${GPU}"
    echo "============================================================"

    if has_acc "$COCOOP_LOG"; then
      echo "[SKIP] CoCoOp done: $COCOOP_LOG"
    else
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/05b_test_cocoop_xd_only.sh" "$SOURCE" "$SEED" "$TARGET"
    fi

    if has_acc "$PRIOR_LOG"; then
      echo "[SKIP] PriorRes done: $PRIOR_LOG"
    else
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/06b_test_cocoop_priorres_xd_only.sh" "$SOURCE" "$SEED" "$TARGET"
    fi

    python "$ROOT/scripts/cocoop_priorres/10_summarize_cocoop_xd.py"
  done
done

python "$ROOT/scripts/cocoop_priorres/10_summarize_cocoop_xd.py"
cat "$ROOT/outputs/cocoop_priorres_xd_summary.md"
