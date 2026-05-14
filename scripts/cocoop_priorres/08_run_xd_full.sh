#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/workspace/meta_prompt_1}
GPU=${GPU:-0}
SEEDS=(1 2 3)
ALL=(caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101)
SOURCES=(caltech101 food101 sun397)

for SOURCE in "${SOURCES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TARGETS=()
    for D in "${ALL[@]}"; do
      if [ "$D" != "$SOURCE" ]; then
        TARGETS+=("$D")
      fi
    done

    echo "[XD full] source=$SOURCE seed=$SEED gpu=$GPU targets=${TARGETS[*]}"
    GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/00_extract_dg_source_features_if_missing.sh" "$SOURCE" "$SEED"
    GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/05_run_cocoop_xd.sh" "$SOURCE" "$SEED" "${TARGETS[@]}"
    GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/06_run_cocoop_priorres_xd.sh" "$SOURCE" "$SEED" "${TARGETS[@]}"
  done
done

python "$ROOT/scripts/cocoop_priorres/10_summarize_cocoop_xd.py"
