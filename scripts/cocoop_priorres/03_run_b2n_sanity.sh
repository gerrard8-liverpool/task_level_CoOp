#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/workspace/meta_prompt_1}
GPU=${GPU:-0}
DATASET=${DATASET:-dtd}
SEED=${SEED:-1}

echo "[B2N sanity] dataset=$DATASET seed=$SEED gpu=$GPU"
GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/00_extract_b2n_base_features.sh" "$DATASET" "$SEED"
GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/01_run_cocoop_b2n.sh" "$DATASET" "$SEED"
GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/02_run_cocoop_priorres_b2n.sh" "$DATASET" "$SEED"
python "$ROOT/scripts/cocoop_priorres/09_summarize_cocoop_b2n.py"
