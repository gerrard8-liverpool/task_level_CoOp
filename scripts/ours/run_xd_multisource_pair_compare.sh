#!/usr/bin/env bash
set -euo pipefail

cd /workspace/meta_prompt_1

GPU_PRIORRES=${GPU_PRIORRES:-1}
GPU_COOP=${GPU_COOP:-2}

ALL_DATASETS=(
  oxford_pets
  eurosat
  dtd
  food101
  oxford_flowers
  caltech101
  stanford_cars
  fgvc_aircraft
  ucf101
  sun397
)

# 默认只跑两个最关键 source；也可以在命令后面手动传 source list
if [ "$#" -gt 0 ]; then
  SOURCES=("$@")
else
  SOURCES=(
    food101
    oxford_pets
  )
fi

SEEDS=(1 2 3)

function build_targets() {
  local source="$1"
  local targets=()

  for d in "${ALL_DATASETS[@]}"; do
    if [ "$d" != "$source" ]; then
      targets+=("$d")
    fi
  done

  echo "${targets[@]}"
}

function has_accuracy() {
  local log="$1"
  if [ -f "$log" ] && grep -q "accuracy:" "$log"; then
    return 0
  else
    return 1
  fi
}

function priorres_done() {
  local source="$1"
  local seed="$2"
  shift 2
  local targets=("$@")

  for target in "${targets[@]}"; do
    local log="third_party/CoOp_clean/output/xd/test/${target}/source_${source}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed${seed}/log.txt"
    if ! has_accuracy "$log"; then
      return 1
    fi
  done

  return 0
}

function coop_done() {
  local source="$1"
  local seed="$2"
  shift 2
  local targets=("$@")

  for target in "${targets[@]}"; do
    local log="third_party/CoOp_clean/output/xd/test/${target}/source_${source}/shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed${seed}/log.txt"
    if ! has_accuracy "$log"; then
      return 1
    fi
  done

  return 0
}

echo "============================================================"
echo "[MULTI-SOURCE XD PAIR COMPARE]"
echo "sources      = ${SOURCES[*]}"
echo "gpu_priorres = ${GPU_PRIORRES}"
echo "gpu_coop     = ${GPU_COOP}"
echo "seeds        = ${SEEDS[*]}"
echo "============================================================"

mkdir -p outputs

for SOURCE in "${SOURCES[@]}"; do
  read -r -a TARGETS <<< "$(build_targets "$SOURCE")"

  echo
  echo "############################################################"
  echo "[SOURCE] ${SOURCE}"
  echo "[TARGETS] ${TARGETS[*]}"
  echo "############################################################"

  for SEED in "${SEEDS[@]}"; do
    echo
    echo "============================================================"
    echo "[SOURCE=${SOURCE}] [SEED=${SEED}]"
    echo "============================================================"

    if priorres_done "$SOURCE" "$SEED" "${TARGETS[@]}"; then
      echo "[SKIP] PriorRes already complete: source=${SOURCE}, seed=${SEED}"
    else
      echo "[RUN] PriorRes safe noalt: source=${SOURCE}, seed=${SEED}"
      GPU=${GPU_PRIORRES} TEST_FEATURE_MODE=source \
      bash scripts/ours/run_priorres_xd_safe_noalt.sh \
      "${SOURCE}" "${SEED}" "${TARGETS[@]}"
    fi

    if coop_done "$SOURCE" "$SEED" "${TARGETS[@]}"; then
      echo "[SKIP] CoOp already complete: source=${SOURCE}, seed=${SEED}"
    else
      echo "[RUN] CoOp baseline: source=${SOURCE}, seed=${SEED}"
      GPU=${GPU_COOP} \
      bash scripts/ours/run_coop_xd_m16k16.sh \
      "${SOURCE}" "${SEED}" "${TARGETS[@]}"
    fi
  done

  OUT_MD="outputs/xd_${SOURCE}_full_compare.md"

  echo
  echo "============================================================"
  echo "[SUMMARY] source=${SOURCE}"
  echo "============================================================"

  python scripts/ours/summarize_xd_compare.py "${SOURCE}" "${TARGETS[@]}" > "${OUT_MD}"
  cat "${OUT_MD}"
done

echo
echo "============================================================"
echo "[ALL DONE] Multi-source XD finished."
echo "Now run:"
echo "python scripts/ours/summarize_xd_multisource_compare.py ${SOURCES[*]}"
echo "============================================================"
