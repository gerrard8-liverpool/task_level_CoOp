#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/CoOp_clean:$PROJECT_ROOT/third_party/Dassl.pytorch:${PYTHONPATH:-}"
export UCX_TLS=tcp,self
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4
export PYTHONFAULTHANDLER=1

mkdir -p logs/a100_full_dg

SOURCES=("caltech101" "food101" "sun397")

targets_for_source() {
  local source="$1"
  case "$source" in
    caltech101)
      echo "oxford_pets eurosat dtd food101 oxford_flowers stanford_cars fgvc_aircraft ucf101 sun397"
      ;;
    food101)
      echo "oxford_pets eurosat dtd caltech101 oxford_flowers stanford_cars fgvc_aircraft ucf101 sun397"
      ;;
    sun397)
      echo "oxford_pets eurosat dtd food101 oxford_flowers caltech101 stanford_cars fgvc_aircraft ucf101"
      ;;
    *)
      echo "[ERROR] Unknown source: $source"
      exit 1
      ;;
  esac
}

echo "============================================================"
echo "[A100 FULL DG] Start"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "GPU=$GPU"
echo "============================================================"

echo "============================================================"
echo "[0] Create backbone configs"
echo "============================================================"
bash scripts/backbone_dg/01_create_backbone_configs.sh

echo "============================================================"
echo "[1] RN50 full DG: CoOp + Safe"
echo "============================================================"

for source in "${SOURCES[@]}"; do
  targets=$(targets_for_source "$source")
  for seed in 1 2 3; do
    echo "---------------- RN50 CoOp source=$source seed=$seed ----------------"
    GPU="$GPU" \
    bash scripts/ours/run_coop_xd_m16k16.sh \
      "$source" "$seed" $targets \
      2>&1 | tee "logs/a100_full_dg/rn50_coop_${source}_seed${seed}.log"

    echo "---------------- RN50 Safe source=$source seed=$seed ----------------"
    GPU="$GPU" \
    TEST_FEATURE_MODE=source \
    bash scripts/ours/run_priorres_xd_safe_noalt.sh \
      "$source" "$seed" $targets \
      2>&1 | tee "logs/a100_full_dg/rn50_safe_${source}_seed${seed}.log"
  done
done

echo "============================================================"
echo "[2] Summarize RN50"
echo "============================================================"
mkdir -p outputs/xd_main_tables

python scripts/ours/summarize_xd_multisource_compare.py \
  caltech101 food101 sun397 \
  | tee outputs/xd_main_tables/xd_rn50_full_coop_safe_a100.md

echo "============================================================"
echo "[3] ViT-B/16 full DG: CoOp + Safe"
echo "============================================================"

GPU="$GPU" \
bash scripts/backbone_dg/05_run_backbone_dg_full.sh vit_b16 \
  2>&1 | tee logs/a100_full_dg/vit_b16_full.log

echo "============================================================"
echo "[4] ViT-B/32 full DG: CoOp + Safe"
echo "============================================================"

GPU="$GPU" \
bash scripts/backbone_dg/05_run_backbone_dg_full.sh vit_b32 \
  2>&1 | tee logs/a100_full_dg/vit_b32_full.log

echo "============================================================"
echo "[5] Final result tables"
echo "============================================================"

echo "RN50:"
cat outputs/xd_main_tables/xd_rn50_full_coop_safe_a100.md || true

echo "ViT-B/16:"
cat outputs/xd_main_tables/xd_vit_b16_full_coop_safe.md || true

echo "ViT-B/32:"
cat outputs/xd_main_tables/xd_vit_b32_full_coop_safe.md || true

echo "============================================================"
echo "[DONE] A100 full DG three-backbone experiment finished."
echo "============================================================"
