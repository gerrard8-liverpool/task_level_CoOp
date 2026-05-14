#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/workspace/meta_prompt_1}
GPU=${GPU:-0}
SOURCE=${SOURCE:-caltech101}
SEED=${SEED:-1}
TARGETS=(dtd eurosat)

echo "[XD sanity] source=$SOURCE seed=$SEED targets=${TARGETS[*]} gpu=$GPU"
GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/00_extract_dg_source_features_if_missing.sh" "$SOURCE" "$SEED"
GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/05_run_cocoop_xd.sh" "$SOURCE" "$SEED" "${TARGETS[@]}"
GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/06_run_cocoop_priorres_xd.sh" "$SOURCE" "$SEED" "${TARGETS[@]}"
python "$ROOT/scripts/cocoop_priorres/10_summarize_cocoop_xd.py"
