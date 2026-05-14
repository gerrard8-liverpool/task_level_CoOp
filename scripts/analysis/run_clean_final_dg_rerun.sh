#!/usr/bin/env bash
set -u

# ============================================================
# Clean final XD rerun:
#   CoOp baseline + Safe PriorRes + Legacy PriorRes
# ============================================================

GPU_ID="${GPU_ID:-0}"
CLEAN="${CLEAN:-1}"
BACKUP="${BACKUP:-1}"
RESUME="${RESUME:-0}"

CFG="${CFG:-rn50_ep50}"
SHOTS="${SHOTS:-16}"
LOAD_EPOCH="${LOAD_EPOCH:-50}"

COOP_TAG="nctx16_cscFalse_ctpend"
SAFE_TAG="nctx16_cscFalse_ctpend_safe_noalt"
LEGACY_TAG="nctx16_cscFalse_ctpend_legacy_noalt"

OUTPUT_ROOT="third_party/CoOp_clean/output/xd"

RUN_ID="final_dg_rerun_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="outputs/${RUN_ID}"
BACKUP_ROOT="${RUN_DIR}/backup_old_outputs"

mkdir -p "$RUN_DIR"
mkdir -p "$BACKUP_ROOT"

# Avoid /tmp I/O failure
mkdir -p /workspace/meta_prompt_1/tmp
chmod 777 /workspace/meta_prompt_1/tmp
export TMPDIR=/workspace/meta_prompt_1/tmp
export TEMP=/workspace/meta_prompt_1/tmp
export TMP=/workspace/meta_prompt_1/tmp

SOURCES=(
  caltech101
  food101
  sun397
  oxford_pets
)

TARGETS=(
  caltech101
  dtd
  eurosat
  food101
  oxford_flowers
  oxford_pets
  stanford_cars
  fgvc_aircraft
  ucf101
  sun397
)

SEEDS=(1 2 3)

STATUS_CSV="${RUN_DIR}/run_status.csv"
echo "timestamp,method,source,seed,target,status,exit_code" > "$STATUS_CSV"

echo "============================================================"
echo "[RUN_ID] $RUN_ID"
echo "[RUN_DIR] $RUN_DIR"
echo "[GPU_ID] $GPU_ID"
echo "[CLEAN] $CLEAN"
echo "[BACKUP] $BACKUP"
echo "[RESUME] $RESUME"
echo "============================================================"

echo
echo "[Preflight] Checking scripts..."
for f in \
  scripts/ours/run_coop_xd_m16k16.sh \
  scripts/ours/run_priorres_xd_safe_noalt.sh \
  scripts/ours/run_priorres_xd_legacy_noalt.sh
do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing script: $f"
    exit 1
  fi
  echo "[OK] $f"
done

echo
echo "[Preflight] Checking precision option in PriorRes scripts..."
for f in scripts/ours/run_priorres_xd_safe_noalt.sh scripts/ours/run_priorres_xd_legacy_noalt.sh; do
  if grep -q "TRAINER.COOP.PREC" "$f"; then
    echo "[OK] $f contains TRAINER.COOP.PREC"
  else
    echo "[WARN] $f does not contain TRAINER.COOP.PREC. If test fails with Half/avg_pool2d, add TRAINER.COOP.PREC fp32."
  fi
done

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

get_test_dir() {
  local method="$1"
  local src="$2"
  local seed="$3"
  local tgt="$4"

  if [ "$method" = "coop" ]; then
    echo "${OUTPUT_ROOT}/test/${tgt}/source_${src}/shots_${SHOTS}/CoOp/${CFG}/${COOP_TAG}/seed${seed}"
  elif [ "$method" = "safe" ]; then
    echo "${OUTPUT_ROOT}/test/${tgt}/source_${src}/shots_${SHOTS}/CoOpPriorRes/${CFG}/${SAFE_TAG}/seed${seed}"
  elif [ "$method" = "legacy" ]; then
    echo "${OUTPUT_ROOT}/test/${tgt}/source_${src}/shots_${SHOTS}/CoOpPriorRes/${CFG}/${LEGACY_TAG}/seed${seed}"
  fi
}

get_train_dir() {
  local method="$1"
  local src="$2"
  local seed="$3"

  if [ "$method" = "coop" ]; then
    echo "${OUTPUT_ROOT}/train/source_${src}/shots_${SHOTS}/CoOp/${CFG}/${COOP_TAG}/seed${seed}"
  elif [ "$method" = "safe" ]; then
    echo "${OUTPUT_ROOT}/train/source_${src}/shots_${SHOTS}/CoOpPriorRes/${CFG}/${SAFE_TAG}/seed${seed}"
  elif [ "$method" = "legacy" ]; then
    echo "${OUTPUT_ROOT}/train/source_${src}/shots_${SHOTS}/CoOpPriorRes/${CFG}/${LEGACY_TAG}/seed${seed}"
  fi
}

run_method() {
  local method="$1"
  local src="$2"
  local seed="$3"
  local tgt="$4"

  local test_dir
  test_dir="$(get_test_dir "$method" "$src" "$seed" "$tgt")"
  local test_log="${test_dir}/log.txt"

  if [ "$RESUME" = "1" ] && has_accuracy "$test_log"; then
    echo "[SKIP] $method source=$src seed=$seed target=$tgt already has accuracy"
    echo "$(date '+%F %T'),${method},${src},${seed},${tgt},SKIP,0" >> "$STATUS_CSV"
    return 0
  fi

  echo
  echo "------------------------------------------------------------"
  echo "[RUN] method=$method source=$src seed=$seed target=$tgt"
  echo "------------------------------------------------------------"

  local start_time
  start_time="$(date '+%F %T')"

  set +e
  if [ "$method" = "coop" ]; then
    GPU="$GPU_ID" bash scripts/ours/run_coop_xd_m16k16.sh "$src" "$seed" "$tgt"
  elif [ "$method" = "safe" ]; then
    GPU="$GPU_ID" bash scripts/ours/run_priorres_xd_safe_noalt.sh "$src" "$seed" "$tgt"
  elif [ "$method" = "legacy" ]; then
    GPU="$GPU_ID" bash scripts/ours/run_priorres_xd_legacy_noalt.sh "$src" "$seed" "$tgt"
  else
    echo "[ERROR] Unknown method: $method"
    return 1
  fi
  local code=$?
  set -e

  if [ "$code" -eq 0 ]; then
    echo "[OK] method=$method source=$src seed=$seed target=$tgt"
    echo "${start_time},${method},${src},${seed},${tgt},OK,${code}" >> "$STATUS_CSV"
  else
    echo "[FAIL] method=$method source=$src seed=$seed target=$tgt exit_code=$code"
    echo "${start_time},${method},${src},${seed},${tgt},FAIL,${code}" >> "$STATUS_CSV"
  fi

  return "$code"
}

# ============================================================
# 1. Clean selected train/test outputs
# ============================================================

if [ "$CLEAN" = "1" ]; then
  echo
  echo "============================================================"
  echo "[CLEAN] Cleaning selected old outputs"
  echo "============================================================"

  for src in "${SOURCES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for method in coop safe legacy; do
        train_dir="$(get_train_dir "$method" "$src" "$seed")"
        backup_and_remove "$train_dir"
      done

      for tgt in "${TARGETS[@]}"; do
        if [ "$tgt" = "$src" ]; then
          continue
        fi

        for method in coop safe legacy; do
          test_dir="$(get_test_dir "$method" "$src" "$seed" "$tgt")"
          backup_and_remove "$test_dir"
        done
      done
    done
  done
else
  echo "[CLEAN] skipped because CLEAN=$CLEAN"
fi

# ============================================================
# 2. Run all experiments
# ============================================================

echo
echo "============================================================"
echo "[RUN] Starting final DG rerun"
echo "============================================================"

for src in "${SOURCES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for tgt in "${TARGETS[@]}"; do
      if [ "$tgt" = "$src" ]; then
        continue
      fi

      run_method coop "$src" "$seed" "$tgt"
      run_method safe "$src" "$seed" "$tgt"
      run_method legacy "$src" "$seed" "$tgt"
    done
  done
done

# ============================================================
# 3. Summarize and plot
# ============================================================

echo
echo "============================================================"
echo "[SUMMARY] Generating final tables and figures"
echo "============================================================"

python scripts/analysis/summarize_final_dg_rerun.py \
  --test-root "$OUTPUT_ROOT/test" \
  --train-root "$OUTPUT_ROOT/train" \
  --out-dir "$RUN_DIR/final_summary" \
  --sources "${SOURCES[@]}" \
  --targets "${TARGETS[@]}" \
  --seeds "${SEEDS[@]}"

echo
echo "============================================================"
echo "[DONE]"
echo "Run directory: $RUN_DIR"
echo "Status CSV: $STATUS_CSV"
echo "Summary directory: $RUN_DIR/final_summary"
echo "============================================================"
