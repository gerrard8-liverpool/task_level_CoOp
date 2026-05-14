#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
GPU=${GPU:-0}
DATASET=${DATASET:-dtd}
SEEDS=(1 2 3)

echo "[B2N sanity 3 seeds] dataset=$DATASET gpu=$GPU"

for SEED in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "[B2N sanity] dataset=$DATASET seed=$SEED gpu=$GPU"
  echo "============================================================"

  GPU="$GPU" DATASET="$DATASET" SEED="$SEED" \
    bash "$ROOT/scripts/cocoop_priorres/03_run_b2n_sanity.sh"
done

python "$ROOT/scripts/cocoop_priorres/09_summarize_cocoop_b2n.py"
cat "$ROOT/outputs/cocoop_priorres_b2n_summary.md"
