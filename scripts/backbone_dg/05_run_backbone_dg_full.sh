#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPU=0 bash scripts/backbone_dg/05_run_backbone_dg_full.sh rn101
#   GPU=0 bash scripts/backbone_dg/05_run_backbone_dg_full.sh vit_b16

BACKBONE_TAG=${1:?Usage: bash 05_run_backbone_dg_full.sh <rn101|vit_b16|vit_b32>}
GPU=${GPU:-0}
ROOT=${ROOT:-/workspace/meta_prompt_1}

cd "$ROOT"

bash scripts/backbone_dg/01_create_backbone_configs.sh
GPU=$GPU bash scripts/backbone_dg/02_extract_backbone_task_features.sh "$BACKBONE_TAG" caltech101 food101 sun397

run_source() {
  local source=$1
  shift
  local targets=("$@")
  for seed in 1 2 3; do
    GPU=$GPU bash scripts/backbone_dg/03_run_backbone_xd_one.sh coop "$BACKBONE_TAG" "$source" "$seed" "${targets[@]}"
    GPU=$GPU bash scripts/backbone_dg/03_run_backbone_xd_one.sh safe "$BACKBONE_TAG" "$source" "$seed" "${targets[@]}"
  done
}

run_source caltech101 oxford_pets eurosat dtd food101 oxford_flowers stanford_cars fgvc_aircraft ucf101 sun397
run_source food101 oxford_pets eurosat dtd caltech101 oxford_flowers stanford_cars fgvc_aircraft ucf101 sun397
run_source sun397 oxford_pets eurosat dtd food101 caltech101 oxford_flowers stanford_cars fgvc_aircraft ucf101

python scripts/backbone_dg/06_summarize_backbone_dg.py "$BACKBONE_TAG" --sources caltech101 food101 sun397 \
  | tee "outputs/xd_main_tables/xd_${BACKBONE_TAG}_full_coop_safe.md"
