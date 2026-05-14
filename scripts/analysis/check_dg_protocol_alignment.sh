#!/usr/bin/env bash
set -euo pipefail

echo "===== Check old runner scripts ====="

echo
echo "[1] CoOp runner usage and defaults"
grep -nE "SOURCE=|SEED=|TARGETS=|SHOTS=|NCTX=|CTX_POS=|CSC=|TRAIN_CFG=|TRAIN_CFG_TAG=|LOAD_EPOCH=|TRAIN_DIR=|for TARGET|--trainer CoOp|DATASET.NUM_SHOTS|DATASET.SUBSAMPLE_CLASSES" \
  scripts/ours/run_coop_xd_m16k16.sh

echo
echo "[2] Safe runner strict DG feature mode"
grep -nE "TEST_FEATURE_MODE|SOURCE_FEATURE_JSON|TEST_FEATURE_JSON|USE_LEGACY_RESIDUAL|USE_CONTEXT_GATING|USE_B|ALTERNATE_OPT|META_LR_RATIO|LAMBDA_MAX|WARMUP_EPOCHS|RAMP_EPOCHS|TRAINER.COOP.PREC" \
  scripts/ours/run_priorres_xd_safe_noalt.sh || true

echo
echo "[3] Legacy runner strict DG feature mode"
grep -nE "TEST_FEATURE_MODE|SOURCE_FEATURE_JSON|TEST_FEATURE_JSON|USE_LEGACY_RESIDUAL|USE_CONTEXT_GATING|USE_B|ALTERNATE_OPT|META_LR_RATIO|LAMBDA_MAX|WARMUP_EPOCHS|RAMP_EPOCHS|TRAINER.COOP.PREC" \
  scripts/ours/run_priorres_xd_legacy_noalt.sh || true

echo
echo "[4] Official old multisource runner"
if [ -f scripts/ours/run_xd_multisource_pair_compare.sh ]; then
  grep -nE "ALL_DATASETS|SOURCES|SEEDS|TEST_FEATURE_MODE=source|run_priorres_xd_safe_noalt|run_coop_xd_m16k16|build_targets" \
    scripts/ours/run_xd_multisource_pair_compare.sh
else
  echo "[WARN] scripts/ours/run_xd_multisource_pair_compare.sh not found"
fi

echo
echo "===== Recommended protocol ====="
cat <<'EOF'
- sources: caltech101, food101, sun397 as main; oxford_pets as source-dependency negative case
- datasets: oxford_pets, eurosat, dtd, food101, oxford_flowers, caltech101, stanford_cars, fgvc_aircraft, ucf101, sun397
- seeds: 1, 2, 3
- train once on source per method/source/seed
- evaluate all targets except source itself
- CoOp: rn50_ep50, shots=16, nctx=16, CSC=False, class token=end
- Precision: follow original config/default; do NOT force fp32 for final protocol rerun
- Safe: CoOpPriorRes safe_noalt, USE_B=False, ALTERNATE_OPT=False, META_LR_RATIO=0.3
- Safe/Legacy DG test feature: TEST_FEATURE_MODE=source
- paths:
  CoOp: output/xd/{train,test}/.../CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed*
  Safe: output/xd/{train,test}/.../CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed*
  Legacy: output/xd/{train,test}/.../CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_legacy_noalt/seed*
EOF
