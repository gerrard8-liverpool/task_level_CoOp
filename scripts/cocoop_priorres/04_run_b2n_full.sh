#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/workspace/meta_prompt_1}
GPU=${GPU:-0}
DATASETS=(caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101)
SEEDS=(1 2 3)

for D in "${DATASETS[@]}"; do
  for S in "${SEEDS[@]}"; do
    echo "[B2N full] dataset=$D seed=$S gpu=$GPU"
    GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/00_extract_b2n_base_features.sh" "$D" "$S"
    GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/01_run_cocoop_b2n.sh" "$D" "$S"
    GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/02_run_cocoop_priorres_b2n.sh" "$D" "$S"
  done
done

python "$ROOT/scripts/cocoop_priorres/09_summarize_cocoop_b2n.py"
