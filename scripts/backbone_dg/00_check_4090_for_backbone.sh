#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPU=0 bash scripts/backbone_dg/00_check_4090_for_backbone.sh

GPU=${GPU:-0}

echo "===== nvidia-smi selected GPU ====="
nvidia-smi -i "$GPU" || true

echo

echo "===== torch cuda check ====="
CUDA_VISIBLE_DEVICES=$GPU python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device_count visible:', torch.cuda.device_count())
    print('device name:', torch.cuda.get_device_name(0))
    prop = torch.cuda.get_device_properties(0)
    print('total memory GB:', round(prop.total_memory / 1024**3, 2))
    print('capability:', prop.major, prop.minor)
PY

echo

echo "If this shows RTX 4090 with about 24GB memory, RN101 should run safely and ViT-B/16 should usually run."
echo "If ViT-B/16 OOMs during evaluation, create a vit_b16_ep50_bs50.yaml fallback with TEST.BATCH_SIZE=50."
