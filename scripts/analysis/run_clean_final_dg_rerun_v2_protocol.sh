#!/usr/bin/env bash
set -u

GPU_ID="${GPU_ID:-0}"
CLEAN="${CLEAN:-1}"
BACKUP="${BACKUP:-1}"
RESUME="${RESUME:-0}"

OUTPUT_ROOT="third_party/CoOp_clean/output/xd"
RUN_ID="final_dg_protocol_rerun_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="outputs/${RUN_ID}"
BACKUP_ROOT="${RUN_DIR}/backup_old_outputs"

mkdir -p "$RUN_DIR" "$BACKUP_ROOT"

mkdir -p /workspace/meta_prompt_1/tmp
chmod 777 /workspace/meta_prompt_1/tmp
export TMPDIR=/workspace/meta_prompt_1/tmp
export TEMP=/workspace/meta_prompt_1/tmp
export TMP=/workspace/meta_prompt_1/tmp

# Old protocol sources:
# main sources + OxfordPets for source-dependency analysis.
SOURCES=(
  caltech101
  food101
  sun397
  oxford_pets
)

# Old CoOp non-ImageNet 10 datasets.
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

SEEDS=(1 2 3)

STATUS_CSV="${RUN_DIR}/run_status.csv"
echo "timestamp,method,source,seed,targets,status,exit_code" > "$STATUS_CSV"

echo "============================================================"
echo "[RUN_ID] $RUN_ID"
echo "[RUN_DIR] $RUN_DIR"
echo "[GPU_ID] $GPU_ID"
echo "[CLEAN] $CLEAN"
echo "[BACKUP] $BACKUP"
echo "[RESUME] $RESUME"
echo "============================================================"

build_targets() {
  local source="$1"
  local targets=()

  for d in "${ALL_DATASETS[@]}"; do
    if [ "$d" != "$source" ]; then
      targets+=("$d")
    fi
  done

  echo "${targets[@]}"
}

backup_and_remove() {
  local path="$1"

  if [ ! -e "$path" ]; then
    return 0
  fi

  if [ "$BACKUP" = "1" ]; then
    local dest="${BACKUP_ROOT}/${path}"
    mkdir -p "$(dirname "$dest")"
    mv "$path" "$dest"
    echo "[BACKUP+REMOVE] $path -> $dest"
  else
    rm -rf "$path"
    echo "[REMOVE] $path"
  fi
}

has_accuracy() {
  local log="$1"
  [ -f "$log" ] && grep -q "accuracy:" "$log"
}

coop_train_dir() {
  local src="$1"
  local seed="$2"
  echo "${OUTPUT_ROOT}/train/source_${src}/shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed${seed}"
}

safe_train_dir() {
  local src="$1"
  local seed="$2"
  echo "${OUTPUT_ROOT}/train/source_${src}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed${seed}"
}

legacy_train_dir() {
  local src="$1"
  local seed="$2"
  echo "${OUTPUT_ROOT}/train/source_${src}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_legacy_noalt/seed${seed}"
}

coop_test_dir() {
  local src="$1"
  local seed="$2"
  local tgt="$3"
  echo "${OUTPUT_ROOT}/test/${tgt}/source_${src}/shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed${seed}"
}

safe_test_dir() {
  local src="$1"
  local seed="$2"
  local tgt="$3"
  echo "${OUTPUT_ROOT}/test/${tgt}/source_${src}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed${seed}"
}

legacy_test_dir() {
  local src="$1"
  local seed="$2"
  local tgt="$3"
  echo "${OUTPUT_ROOT}/test/${tgt}/source_${src}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_legacy_noalt/seed${seed}"
}

method_done() {
  local method="$1"
  local src="$2"
  local seed="$3"
  shift 3
  local targets=("$@")

  for tgt in "${targets[@]}"; do
    local log=""
    if [ "$method" = "coop" ]; then
      log="$(coop_test_dir "$src" "$seed" "$tgt")/log.txt"
    elif [ "$method" = "safe" ]; then
      log="$(safe_test_dir "$src" "$seed" "$tgt")/log.txt"
    else
      log="$(legacy_test_dir "$src" "$seed" "$tgt")/log.txt"
    fi

    if ! has_accuracy "$log"; then
      return 1
    fi
  done

  return 0
}

clean_source_seed_method() {
  local method="$1"
  local src="$2"
  local seed="$3"
  shift 3
  local targets=("$@")

  if [ "$method" = "coop" ]; then
    backup_and_remove "$(coop_train_dir "$src" "$seed")"
    for tgt in "${targets[@]}"; do
      backup_and_remove "$(coop_test_dir "$src" "$seed" "$tgt")"
    done
  elif [ "$method" = "safe" ]; then
    backup_and_remove "$(safe_train_dir "$src" "$seed")"
    for tgt in "${targets[@]}"; do
      backup_and_remove "$(safe_test_dir "$src" "$seed" "$tgt")"
    done
  elif [ "$method" = "legacy" ]; then
    backup_and_remove "$(legacy_train_dir "$src" "$seed")"
    for tgt in "${targets[@]}"; do
      backup_and_remove "$(legacy_test_dir "$src" "$seed" "$tgt")"
    done
  fi
}

run_source_seed_method() {
  local method="$1"
  local src="$2"
  local seed="$3"
  shift 3
  local targets=("$@")
  local target_str="${targets[*]}"

  if [ "$RESUME" = "1" ] && method_done "$method" "$src" "$seed" "${targets[@]}"; then
    echo "[SKIP] method=$method source=$src seed=$seed all target logs complete"
    echo "$(date '+%F %T'),${method},${src},${seed},\"${target_str}\",SKIP,0" >> "$STATUS_CSV"
    return 0
  fi

  echo
  echo "============================================================"
  echo "[RUN] method=$method source=$src seed=$seed"
  echo "[TARGETS] ${targets[*]}"
  echo "============================================================"

  local code=0
  set +e

  if [ "$method" = "coop" ]; then
    GPU="$GPU_ID" \
    bash scripts/ours/run_coop_xd_m16k16.sh "$src" "$seed" "${targets[@]}"
    code=$?
  elif [ "$method" = "safe" ]; then
    GPU="$GPU_ID" TEST_FEATURE_MODE=source \
    bash scripts/ours/run_priorres_xd_safe_noalt.sh "$src" "$seed" "${targets[@]}"
    code=$?
  elif [ "$method" = "legacy" ]; then
    GPU="$GPU_ID" TEST_FEATURE_MODE=source \
    bash scripts/ours/run_priorres_xd_legacy_noalt.sh "$src" "$seed" "${targets[@]}"
    code=$?
  fi

  set -e

  if [ "$code" -eq 0 ]; then
    echo "[OK] method=$method source=$src seed=$seed"
    echo "$(date '+%F %T'),${method},${src},${seed},\"${target_str}\",OK,0" >> "$STATUS_CSV"
  else
    echo "[FAIL] method=$method source=$src seed=$seed code=$code"
    echo "$(date '+%F %T'),${method},${src},${seed},\"${target_str}\",FAIL,${code}" >> "$STATUS_CSV"
  fi

  return "$code"
}

echo
echo "[Preflight] Current protocol:"
echo "SOURCES=${SOURCES[*]}"
echo "ALL_DATASETS=${ALL_DATASETS[*]}"
echo "SEEDS=${SEEDS[*]}"
echo "Safe/Legacy TEST_FEATURE_MODE=source"
echo "RN50, shots=16, nctx=16, CSC=False, class token=end"

if [ "$CLEAN" = "1" ]; then
  echo
  echo "============================================================"
  echo "[CLEAN] Cleaning selected old protocol outputs"
  echo "============================================================"

  for src in "${SOURCES[@]}"; do
    read -r -a targets <<< "$(build_targets "$src")"

    for seed in "${SEEDS[@]}"; do
      clean_source_seed_method coop "$src" "$seed" "${targets[@]}"
      clean_source_seed_method safe "$src" "$seed" "${targets[@]}"
      clean_source_seed_method legacy "$src" "$seed" "${targets[@]}"
    done
  done
else
  echo "[CLEAN] skipped"
fi

echo
echo "============================================================"
echo "[RUN] Protocol-aligned final DG rerun"
echo "============================================================"

for src in "${SOURCES[@]}"; do
  read -r -a targets <<< "$(build_targets "$src")"

  for seed in "${SEEDS[@]}"; do
    run_source_seed_method coop "$src" "$seed" "${targets[@]}"
    run_source_seed_method safe "$src" "$seed" "${targets[@]}"
    run_source_seed_method legacy "$src" "$seed" "${targets[@]}"
  done
done

echo
echo "============================================================"
echo "[SUMMARY]"
echo "============================================================"

python scripts/analysis/summarize_final_dg_rerun.py \
  --test-root "$OUTPUT_ROOT/test" \
  --train-root "$OUTPUT_ROOT/train" \
  --out-dir "$RUN_DIR/final_summary" \
  --sources "${SOURCES[@]}" \
  --targets "${ALL_DATASETS[@]}" \
  --seeds "${SEEDS[@]}"

echo
echo "[DONE] Protocol-aligned run finished."
echo "RUN_DIR=$RUN_DIR"
echo "STATUS_CSV=$STATUS_CSV"
echo "SUMMARY=$RUN_DIR/final_summary"
