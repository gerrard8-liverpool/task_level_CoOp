#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPU=0 bash scripts/backbone_dg/04_run_backbone_dg_sanity.sh rn101
#   GPU=0 bash scripts/backbone_dg/04_run_backbone_dg_sanity.sh vit_b16

BACKBONE_TAG=${1:?Usage: bash 04_run_backbone_dg_sanity.sh <rn101|vit_b16|vit_b32>}
GPU=${GPU:-0}
ROOT=${ROOT:-/workspace/meta_prompt_1}

cd "$ROOT"

bash scripts/backbone_dg/01_create_backbone_configs.sh
GPU=$GPU bash scripts/backbone_dg/02_extract_backbone_task_features.sh "$BACKBONE_TAG" caltech101

for SEED in 1 2 3; do
  GPU=$GPU bash scripts/backbone_dg/03_run_backbone_xd_one.sh coop "$BACKBONE_TAG" caltech101 "$SEED" eurosat dtd sun397
  GPU=$GPU bash scripts/backbone_dg/03_run_backbone_xd_one.sh safe "$BACKBONE_TAG" caltech101 "$SEED" eurosat dtd sun397
done

python scripts/backbone_dg/06_summarize_backbone_dg.py "$BACKBONE_TAG" --sources caltech101 --targets eurosat dtd sun397 \
  | tee "outputs/xd_main_tables/xd_${BACKBONE_TAG}_sanity_coop_safe.md"
