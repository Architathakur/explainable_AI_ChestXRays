#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/raw/rsna}"
WEIGHTS="${WEIGHTS:-densenet121-res224-chex}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
FINE_TUNE_EPOCHS="${FINE_TUNE_EPOCHS:-0}"
FINE_TUNE_TRAIN_LIMIT="${FINE_TUNE_TRAIN_LIMIT:-0}"
FINE_TUNE_VAL_LIMIT="${FINE_TUNE_VAL_LIMIT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cxr_foundation_${WEIGHTS}}"
CACHE_DIR="${CACHE_DIR:-outputs/feature_cache}"

mkdir -p outputs/training_logs

EXTRA_ARGS=()
if [[ "$FINE_TUNE_TRAIN_LIMIT" != "0" ]]; then
  EXTRA_ARGS+=(--fine_tune_train_limit "$FINE_TUNE_TRAIN_LIMIT")
fi
if [[ "$FINE_TUNE_VAL_LIMIT" != "0" ]]; then
  EXTRA_ARGS+=(--fine_tune_val_limit "$FINE_TUNE_VAL_LIMIT")
fi

PYTHONPATH=. .venv-train/bin/python xai_pneumonia/train_cxr_foundation.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --weights "$WEIGHTS" \
  --batch_size "$BATCH_SIZE" \
  --feature_batch_size "$FEATURE_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --fine_tune_epochs "$FINE_TUNE_EPOCHS" \
  --cache_dir "$CACHE_DIR" \
  ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"} \
  2>&1 | tee "outputs/training_logs/cxr_foundation_${WEIGHTS}_$(date +%Y%m%d_%H%M%S).log"
