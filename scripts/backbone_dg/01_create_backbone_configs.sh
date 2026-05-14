#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
CFG_DIR="$COOP_ROOT/configs/trainers/CoOp"
BASE_CFG="$CFG_DIR/rn50_ep50.yaml"

if [ ! -f "$BASE_CFG" ]; then
  echo "[ERROR] Missing base config: $BASE_CFG"
  exit 1
fi

python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ.get("COOP_ROOT", "/workspace/meta_prompt_1/third_party/CoOp_clean"))
cfg_dir = root / "configs/trainers/CoOp"
base = cfg_dir / "rn50_ep50.yaml"
text = base.read_text()

configs = {
    "rn101_ep50.yaml": "RN101",
    "vit_b16_ep50.yaml": "ViT-B/16",
    "vit_b32_ep50.yaml": "ViT-B/32",
}

for name, backbone in configs.items():
    out = cfg_dir / name
    new_text = text.replace('NAME: "RN50"', f'NAME: "{backbone}"')
    if 'NAME: "RN50"' not in text:
        raise RuntimeError("Cannot find MODEL.BACKBONE.NAME RN50 line in rn50_ep50.yaml")
    out.write_text(new_text)
    print(f"[WROTE] {out} with MODEL.BACKBONE.NAME={backbone}")
PY

ls -lh "$CFG_DIR"/*_ep50.yaml
