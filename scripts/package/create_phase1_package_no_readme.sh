#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/meta_prompt_1"
cd "$PROJECT_ROOT"

TS=$(date +%Y%m%d_%H%M%S)
PKG_ROOT="outputs/final_phase1_package_${TS}"
ZIP_PATH="outputs/final_phase1_package_${TS}.zip"

mkdir -p "$PKG_ROOT"

mkdir -p "$PKG_ROOT/results/dg_main"
mkdir -p "$PKG_ROOT/results/safe_prior_ablation"
mkdir -p "$PKG_ROOT/results/b2n"
mkdir -p "$PKG_ROOT/results/indomain"
mkdir -p "$PKG_ROOT/results/residual_formula"

mkdir -p "$PKG_ROOT/scripts/dg"
mkdir -p "$PKG_ROOT/scripts/safe_prior_ablation"
mkdir -p "$PKG_ROOT/scripts/b2n"
mkdir -p "$PKG_ROOT/scripts/package_and_check"

mkdir -p "$PKG_ROOT/task_features"
mkdir -p "$PKG_ROOT/mechanism_checks"
mkdir -p "$PKG_ROOT/code_snapshot/trainers"
mkdir -p "$PKG_ROOT/code_snapshot/meta_prompts"
mkdir -p "$PKG_ROOT/configs/datasets"
mkdir -p "$PKG_ROOT/configs/trainers"

MISSING="$PKG_ROOT/missing_files_report.txt"
: > "$MISSING"

copy_if_exists() {
  local src="$1"
  local dst="$2"

  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
    echo "[OK] $src -> $dst"
  else
    echo "[MISSING] $src" | tee -a "$MISSING"
  fi
}

copy_glob_if_exists() {
  local pattern="$1"
  local dst_dir="$2"
  local found=0

  mkdir -p "$dst_dir"

  shopt -s nullglob
  for f in $pattern; do
    cp -a "$f" "$dst_dir/"
    echo "[OK] $f -> $dst_dir/"
    found=1
  done
  shopt -u nullglob

  if [ "$found" -eq 0 ]; then
    echo "[MISSING_GLOB] $pattern" | tee -a "$MISSING"
  fi
}

echo "============================================================"
echo "[1/8] Copy main result tables"
echo "============================================================"

copy_if_exists \
  "outputs/xd_main_tables/xd_multisource_coop_safe_legacy.md" \
  "$PKG_ROOT/results/dg_main/xd_multisource_coop_safe_legacy.md"

copy_if_exists \
  "outputs/ablations/safe_prior/tables/safe_prior_xd_ablation.md" \
  "$PKG_ROOT/results/safe_prior_ablation/safe_prior_xd_ablation.md"

if [ -f "outputs/b2n_tables/b2n_compare_safe_priorres_vs_coop.md" ]; then
  copy_if_exists \
    "outputs/b2n_tables/b2n_compare_safe_priorres_vs_coop.md" \
    "$PKG_ROOT/results/b2n/b2n_compare_safe_priorres_vs_coop.md"
elif [ -f "outputs/b2n_compare_safe_priorres_vs_coop.md" ]; then
  copy_if_exists \
    "outputs/b2n_compare_safe_priorres_vs_coop.md" \
    "$PKG_ROOT/results/b2n/b2n_compare_safe_priorres_vs_coop.md"
elif [ -f "Result/b2n_compare_safe_priorres_vs_coop.md" ]; then
  copy_if_exists \
    "Result/b2n_compare_safe_priorres_vs_coop.md" \
    "$PKG_ROOT/results/b2n/b2n_compare_safe_priorres_vs_coop.md"
else
  found_b2n=$(find . -path "*b2n_compare_safe_priorres_vs_coop.md" -type f | head -1 || true)
  if [ -n "$found_b2n" ]; then
    copy_if_exists \
      "$found_b2n" \
      "$PKG_ROOT/results/b2n/b2n_compare_safe_priorres_vs_coop.md"
  else
    echo "[MISSING] b2n_compare_safe_priorres_vs_coop.md" | tee -a "$MISSING"
  fi
fi

if [ -f "outputs/main_compare_safe_vs_coop.csv" ]; then
  copy_if_exists \
    "outputs/main_compare_safe_vs_coop.csv" \
    "$PKG_ROOT/results/indomain/main_compare_safe_vs_coop.csv"
elif [ -f "Result/main_compare_safe_vs_coop.csv" ]; then
  copy_if_exists \
    "Result/main_compare_safe_vs_coop.csv" \
    "$PKG_ROOT/results/indomain/main_compare_safe_vs_coop.csv"
else
  found_indomain=$(find . -path "*main_compare_safe_vs_coop.csv" -type f | head -1 || true)
  if [ -n "$found_indomain" ]; then
    copy_if_exists \
      "$found_indomain" \
      "$PKG_ROOT/results/indomain/main_compare_safe_vs_coop.csv"
  else
    echo "[MISSING_OPTIONAL] main_compare_safe_vs_coop.csv" | tee -a "$MISSING"
  fi
fi

copy_glob_if_exists \
  "outputs/ablations/residual_formula/tables/*.md" \
  "$PKG_ROOT/results/residual_formula"

echo "============================================================"
echo "[2/8] Copy DG scripts"
echo "============================================================"

copy_if_exists \
  "scripts/ours/run_coop_xd_m16k16.sh" \
  "$PKG_ROOT/scripts/dg/run_coop_xd_m16k16.sh"

copy_if_exists \
  "scripts/ours/run_priorres_xd_safe_noalt.sh" \
  "$PKG_ROOT/scripts/dg/run_priorres_xd_safe_noalt.sh"

copy_if_exists \
  "scripts/ours/run_priorres_xd_legacy_noalt.sh" \
  "$PKG_ROOT/scripts/dg/run_priorres_xd_legacy_noalt.sh"

copy_if_exists \
  "scripts/ours/summarize_xd_multisource_compare.py" \
  "$PKG_ROOT/scripts/dg/summarize_xd_multisource_compare.py"

copy_if_exists \
  "scripts/ours/summarize_xd_multisource_compare_with_legacy.py" \
  "$PKG_ROOT/scripts/dg/summarize_xd_multisource_compare_with_legacy.py"

copy_if_exists \
  "scripts/ours/run_xd_multisource_pair_compare.sh" \
  "$PKG_ROOT/scripts/dg/run_xd_multisource_pair_compare.sh"

echo "============================================================"
echo "[3/8] Copy safe prior ablation scripts"
echo "============================================================"

copy_if_exists \
  "scripts/ablations/safe_prior/run_safe_prior_xd.sh" \
  "$PKG_ROOT/scripts/safe_prior_ablation/run_safe_prior_xd.sh"

copy_if_exists \
  "scripts/ablations/safe_prior/summarize_safe_prior_xd.py" \
  "$PKG_ROOT/scripts/safe_prior_ablation/summarize_safe_prior_xd.py"

echo "============================================================"
echo "[4/8] Copy B2N scripts"
echo "============================================================"

copy_if_exists \
  "scripts/ours/run_coop_b2n_m16k16.sh" \
  "$PKG_ROOT/scripts/b2n/run_coop_b2n_m16k16.sh"

copy_if_exists \
  "scripts/ours/run_priorres_b2n_safe_noalt.sh" \
  "$PKG_ROOT/scripts/b2n/run_priorres_b2n_safe_noalt.sh"

copy_if_exists \
  "scripts/ours/summarize_b2n_compare.py" \
  "$PKG_ROOT/scripts/b2n/summarize_b2n_compare.py"

echo "============================================================"
echo "[5/8] Copy task features"
echo "============================================================"

copy_glob_if_exists \
  "outputs/task_features/*_train.json" \
  "$PKG_ROOT/task_features"

copy_if_exists \
  "outputs/task_features/mean_train_feature.json" \
  "$PKG_ROOT/task_features/mean_train_feature.json"

copy_if_exists \
  "outputs/task_features/imagenet_train_sample32.json" \
  "$PKG_ROOT/task_features/imagenet_train_sample32.json"

echo "============================================================"
echo "[6/8] Copy mechanism logs"
echo "============================================================"

copy_glob_if_exists \
  "outputs/ablations/mechanism_checks/*.txt" \
  "$PKG_ROOT/mechanism_checks"

echo "============================================================"
echo "[7/8] Copy key code snapshot and configs"
echo "============================================================"

copy_if_exists \
  "third_party/CoOp_clean/trainers/coop_priorres.py" \
  "$PKG_ROOT/code_snapshot/trainers/coop_priorres.py"

copy_if_exists \
  "third_party/CoOp_clean/trainers/coop.py" \
  "$PKG_ROOT/code_snapshot/trainers/coop.py"

copy_glob_if_exists \
  "src/meta_prompts/*.py" \
  "$PKG_ROOT/code_snapshot/meta_prompts"

copy_glob_if_exists \
  "third_party/CoOp_clean/configs/datasets/*.yaml" \
  "$PKG_ROOT/configs/datasets"

copy_if_exists \
  "third_party/CoOp_clean/configs/trainers/CoOp/rn50_ep50.yaml" \
  "$PKG_ROOT/configs/trainers/rn50_ep50.yaml"

echo "============================================================"
echo "[8/8] Write checksums, protocol report, zip"
echo "============================================================"

{
  echo "# Protocol Check"
  echo
  echo "DG / safe-prior scripts should not contain fp32 or explicit batch-size overrides."
  echo
  grep -RIn "TRAINER.COOP.PREC\\|DATALOADER.TRAIN_X.BATCH_SIZE\\|DATALOADER.TEST.BATCH_SIZE\\|TRAIN_BS\\|TEST_BS" \
    "$PKG_ROOT/scripts/dg" "$PKG_ROOT/scripts/safe_prior_ablation" 2>/dev/null || true
} > "$PKG_ROOT/protocol_check_report.txt"

find "$PKG_ROOT" -type f | sort > "$PKG_ROOT/package_file_list.txt"

(
  cd "$PKG_ROOT"
  find . -type f | sort | xargs sha256sum
) > "$PKG_ROOT/sha256sum.txt"

cd outputs
zip -r "$(basename "$ZIP_PATH")" "$(basename "$PKG_ROOT")" >/dev/null
cd "$PROJECT_ROOT"

echo
echo "============================================================"
echo "PACKAGE CREATED"
echo "============================================================"
echo "Directory: $PKG_ROOT"
echo "Zip:       $ZIP_PATH"
echo
echo "Missing report:"
cat "$MISSING"
echo
echo "Protocol check report:"
cat "$PKG_ROOT/protocol_check_report.txt"
echo
echo "File list:"
echo "$PKG_ROOT/package_file_list.txt"
echo "============================================================"
