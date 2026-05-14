#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
GPU=${GPU:-0}
SOURCE=${SOURCE:-caltech101}
SEEDS=(1 2 3)

# 默认 sanity target：DTD + EuroSAT
# DTD: texture shift / B2N sanity 已有正信号
# EuroSAT: domain shift 明显，之前 DG 主表里很敏感
TARGETS=(dtd eurosat)

echo "[XD sanity 3 seeds] source=$SOURCE targets=${TARGETS[*]} gpu=$GPU"

for SEED in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "[XD sanity] source=$SOURCE seed=$SEED gpu=$GPU targets=${TARGETS[*]}"
  echo "============================================================"

  GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/00_extract_dg_source_features_if_missing.sh" "$SOURCE" "$SEED"
  GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/05_run_cocoop_xd.sh" "$SOURCE" "$SEED" "${TARGETS[@]}"
  GPU="$GPU" bash "$ROOT/scripts/cocoop_priorres/06_run_cocoop_priorres_xd.sh" "$SOURCE" "$SEED" "${TARGETS[@]}"
done

python "$ROOT/scripts/cocoop_priorres/10_summarize_cocoop_xd.py"
cat "$ROOT/outputs/cocoop_priorres_xd_summary.md"
