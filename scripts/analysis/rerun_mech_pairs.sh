#!/usr/bin/env bash
set -e

GPU_ID="${GPU_ID:-0}"
BACKUP_ROOT="outputs/backup_before_mechanism_figs/rerun_mech_pairs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_ROOT"

pairs=(
  "caltech101 1 eurosat"
  "caltech101 2 eurosat"
  "food101 1 oxford_pets"
  "sun397 1 eurosat"
)

for item in "${pairs[@]}"; do
  set -- $item
  SRC="$1"
  SEED="$2"
  TGT="$3"

  echo
  echo "============================================================"
  echo "[PAIR] source=$SRC seed=$SEED target=$TGT"
  echo "============================================================"

  SAFE_DIR="third_party/CoOp_clean/output/xd/train/source_${SRC}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed${SEED}"
  LEGACY_DIR="third_party/CoOp_clean/output/xd/train/source_${SRC}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_legacy_noalt/seed${SEED}"

  SAFE_TEST="third_party/CoOp_clean/output/xd/test/${TGT}/source_${SRC}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed${SEED}"
  LEGACY_TEST="third_party/CoOp_clean/output/xd/test/${TGT}/source_${SRC}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_legacy_noalt/seed${SEED}"

  mkdir -p "$BACKUP_ROOT/source_${SRC}_seed${SEED}"

  if [ -d "$SAFE_DIR" ]; then
    cp -a "$SAFE_DIR" "$BACKUP_ROOT/source_${SRC}_seed${SEED}/safe_train"
  fi

  if [ -d "$LEGACY_DIR" ]; then
    cp -a "$LEGACY_DIR" "$BACKUP_ROOT/source_${SRC}_seed${SEED}/legacy_train"
  fi

  if [ -d "$SAFE_TEST" ]; then
    cp -a "$SAFE_TEST" "$BACKUP_ROOT/source_${SRC}_seed${SEED}/safe_test_${TGT}"
  fi

  if [ -d "$LEGACY_TEST" ]; then
    cp -a "$LEGACY_TEST" "$BACKUP_ROOT/source_${SRC}_seed${SEED}/legacy_test_${TGT}"
  fi

  rm -rf "$SAFE_DIR" "$LEGACY_DIR" "$SAFE_TEST" "$LEGACY_TEST"

  echo "[RUN] Safe PriorRes"
  GPU="$GPU_ID" bash scripts/ours/run_priorres_xd_safe_noalt.sh "$SRC" "$SEED" "$TGT"

  echo "[RUN] Legacy PriorRes"
  GPU="$GPU_ID" bash scripts/ours/run_priorres_xd_legacy_noalt.sh "$SRC" "$SEED" "$TGT"

  echo "[CHECK] analysis_stats.csv"
  ls "$SAFE_DIR/analysis_stats.csv"
  ls "$LEGACY_DIR/analysis_stats.csv"

  echo "[ACC] Safe"
  grep -n "accuracy:" "$SAFE_TEST/log.txt" || true

  echo "[ACC] Legacy"
  grep -n "accuracy:" "$LEGACY_TEST/log.txt" || true
done

echo
echo "[DONE] backups saved to: $BACKUP_ROOT"
