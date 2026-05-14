#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$PROJECT_ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}

SOURCE=${SOURCE:-imagenet}
SEEDS_STR=${SEEDS:-"1 2 3"}
METHODS_STR=${METHODS:-"cocoop priorres"}

TRAIN_CFG=${TRAIN_CFG:-configs/trainers/CoCoOp/rn50_c4_ep10_batch4_a100.yaml}
TRAIN_CFG_TAG=${TRAIN_CFG_TAG:-rn50_c4_ep10_batch4_a100}
LOAD_EPOCH=${LOAD_EPOCH:-10}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-4}

FEATURE_JSON=${SOURCE_FEATURE_JSON:-$PROJECT_ROOT/outputs/task_features/imagenet_train_sample32.json}

TARGETS=(
  caltech101
  oxford_pets
  dtd
  eurosat
  food101
  oxford_flowers
  stanford_cars
  fgvc_aircraft
  ucf101
  sun397
)

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$COOP_ROOT:$PROJECT_ROOT/third_party/Dassl.pytorch:${PYTHONPATH:-}"
export UCX_TLS=tcp,self
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONFAULTHANDLER=1

mkdir -p "$PROJECT_ROOT/logs/a100_cocoop_imagenet_dg_resume"

cd "$PROJECT_ROOT"

has_acc() {
  local log="$1"
  [ -f "$log" ] && grep -qE "accuracy:" "$log"
}

cocoop_common() {
  local seed="$1"
  echo "source_${SOURCE}/shots_${SHOTS}/CoCoOp/${TRAIN_CFG_TAG}/nctx${NCTX}_ctxinit_a_photo_of_a/seed${seed}"
}

priorres_common() {
  local seed="$1"
  echo "source_${SOURCE}/shots_${SHOTS}/CoCoOpPriorRes/${TRAIN_CFG_TAG}/nctx${NCTX}_ctxinit_a_photo_of_a_safe_noalt_sourceprior/seed${seed}"
}

clean_bad_cocoop_train_if_needed() {
  local seed="$1"
  local common
  common=$(cocoop_common "$seed")
  local train_dir="$COOP_ROOT/output_cocoop_priorres/xd/train/${common}"
  local ckpt="$train_dir/prompt_learner/model.pth.tar-${LOAD_EPOCH}"

  if [ -f "$ckpt" ]; then
    echo "[OK TRAIN] CoCoOp seed=${seed}"
    return 0
  fi

  if [ -d "$train_dir" ]; then
    echo "[CLEAN BAD TRAIN] CoCoOp seed=${seed}: $train_dir"
    rm -rf "$train_dir"
  fi

  return 1
}

clean_bad_priorres_train_if_needed() {
  local seed="$1"
  local common
  common=$(priorres_common "$seed")
  local train_dir="$COOP_ROOT/output_cocoop_priorres/xd/train/${common}"
  local ckpt1="$train_dir/prompt_learner/model.pth.tar-${LOAD_EPOCH}"
  local ckpt2="$train_dir/prior_adapter/model.pth.tar-${LOAD_EPOCH}"

  if [ -f "$ckpt1" ] && [ -f "$ckpt2" ]; then
    echo "[OK TRAIN] CoCoOpPriorRes seed=${seed}"
    return 0
  fi

  if [ -d "$train_dir" ]; then
    echo "[CLEAN BAD TRAIN] CoCoOpPriorRes seed=${seed}: $train_dir"
    rm -rf "$train_dir"
  fi

  return 1
}

clean_bad_cocoop_test_if_needed() {
  local seed="$1"
  local target="$2"
  local common
  common=$(cocoop_common "$seed")
  local test_dir="$COOP_ROOT/output_cocoop_priorres/xd/test/${target}/${common}"
  local log="$test_dir/log.txt"

  if has_acc "$log"; then
    echo "[OK TEST] CoCoOp ${SOURCE}->${target} seed=${seed}"
    return 0
  fi

  if [ -d "$test_dir" ]; then
    echo "[CLEAN BAD TEST] CoCoOp ${SOURCE}->${target} seed=${seed}: $test_dir"
    rm -rf "$test_dir"
  fi

  return 1
}

clean_bad_priorres_test_if_needed() {
  local seed="$1"
  local target="$2"
  local common
  common=$(priorres_common "$seed")
  local test_dir="$COOP_ROOT/output_cocoop_priorres/xd/test/${target}/${common}"
  local log="$test_dir/log.txt"

  if has_acc "$log"; then
    echo "[OK TEST] CoCoOpPriorRes ${SOURCE}->${target} seed=${seed}"
    return 0
  fi

  if [ -d "$test_dir" ]; then
    echo "[CLEAN BAD TEST] CoCoOpPriorRes ${SOURCE}->${target} seed=${seed}: $test_dir"
    rm -rf "$test_dir"
  fi

  return 1
}

run_cocoop_train() {
  local seed="$1"

  if clean_bad_cocoop_train_if_needed "$seed"; then
    return 0
  fi

  echo "============================================================"
  echo "[RUN TRAIN] CoCoOp source=${SOURCE} seed=${seed}"
  echo "============================================================"

  GPU="$GPU" \
  TRAIN_CFG="$TRAIN_CFG" \
  TRAIN_CFG_TAG="$TRAIN_CFG_TAG" \
  bash scripts/cocoop_priorres/05a_train_cocoop_xd_only.sh \
    "$SOURCE" "$seed" \
    2>&1 | tee "logs/a100_cocoop_imagenet_dg_resume/cocoop_train_seed${seed}.log"
}

run_priorres_train() {
  local seed="$1"

  if [ ! -f "$FEATURE_JSON" ]; then
    echo "[ERROR] Missing source feature: $FEATURE_JSON"
    exit 1
  fi

  if clean_bad_priorres_train_if_needed "$seed"; then
    return 0
  fi

  echo "============================================================"
  echo "[RUN TRAIN] CoCoOpPriorRes source=${SOURCE} seed=${seed}"
  echo "============================================================"

  GPU="$GPU" \
  TRAIN_CFG="$TRAIN_CFG" \
  TRAIN_CFG_TAG="$TRAIN_CFG_TAG" \
  SOURCE_FEATURE_JSON="$FEATURE_JSON" \
  bash scripts/cocoop_priorres/06a_train_cocoop_priorres_xd_only.sh \
    "$SOURCE" "$seed" \
    2>&1 | tee "logs/a100_cocoop_imagenet_dg_resume/priorres_train_seed${seed}.log"
}

run_cocoop_tests() {
  local seed="$1"

  for target in "${TARGETS[@]}"; do
    if clean_bad_cocoop_test_if_needed "$seed" "$target"; then
      continue
    fi

    echo "============================================================"
    echo "[RUN TEST] CoCoOp ${SOURCE}->${target} seed=${seed}"
    echo "============================================================"

    GPU="$GPU" \
    TRAIN_CFG="$TRAIN_CFG" \
    TRAIN_CFG_TAG="$TRAIN_CFG_TAG" \
    bash scripts/cocoop_priorres/05b_test_cocoop_xd_only.sh \
      "$SOURCE" "$seed" "$target" \
      2>&1 | tee "logs/a100_cocoop_imagenet_dg_resume/cocoop_test_${target}_seed${seed}.log"
  done
}

run_priorres_tests() {
  local seed="$1"

  if [ ! -f "$FEATURE_JSON" ]; then
    echo "[ERROR] Missing source feature: $FEATURE_JSON"
    exit 1
  fi

  for target in "${TARGETS[@]}"; do
    if clean_bad_priorres_test_if_needed "$seed" "$target"; then
      continue
    fi

    echo "============================================================"
    echo "[RUN TEST] CoCoOpPriorRes ${SOURCE}->${target} seed=${seed}"
    echo "============================================================"

    GPU="$GPU" \
    TRAIN_CFG="$TRAIN_CFG" \
    TRAIN_CFG_TAG="$TRAIN_CFG_TAG" \
    SOURCE_FEATURE_JSON="$FEATURE_JSON" \
    bash scripts/cocoop_priorres/06b_test_cocoop_priorres_xd_only.sh \
      "$SOURCE" "$seed" "$target" \
      2>&1 | tee "logs/a100_cocoop_imagenet_dg_resume/priorres_test_${target}_seed${seed}.log"
  done
}

echo "============================================================"
echo "[RESUME IMAGENET-SOURCE COCOOP DG]"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "SOURCE=$SOURCE"
echo "SEEDS=$SEEDS_STR"
echo "METHODS=$METHODS_STR"
echo "GPU=$GPU"
echo "TRAIN_CFG=$TRAIN_CFG"
echo "TRAIN_CFG_TAG=$TRAIN_CFG_TAG"
echo "FEATURE_JSON=$FEATURE_JSON"
echo "============================================================"

for seed in $SEEDS_STR; do
  for method in $METHODS_STR; do
    case "$method" in
      cocoop)
        run_cocoop_train "$seed"
        run_cocoop_tests "$seed"
        ;;
      priorres)
        run_priorres_train "$seed"
        run_priorres_tests "$seed"
        ;;
      *)
        echo "[ERROR] Unknown method: $method"
        echo "Use METHODS=\"cocoop priorres\" or METHODS=\"cocoop\" or METHODS=\"priorres\""
        exit 1
        ;;
    esac
  done
done

echo "============================================================"
echo "[DONE] Resume script finished."
echo "============================================================"
