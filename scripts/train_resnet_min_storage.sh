#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/raw/rsna}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-0}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-5}"
FINE_TUNE_LAYERS="${FINE_TUNE_LAYERS:-0}"
FIT_VERBOSE="${FIT_VERBOSE:-2}"

mkdir -p outputs/training_logs

PYTHONPATH=. .venv-train/bin/python xai_pneumonia/train.py \
  --data_dir "$DATA_DIR" \
  --output_dir outputs \
  --epochs "$EPOCHS" \
  --phase1_epochs "$PHASE1_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --fine_tune_layers "$FINE_TUNE_LAYERS" \
  --fit_verbose "$FIT_VERBOSE" \
  --skip_mean_image \
  --skip_layer_selection \
  2>&1 | tee outputs/training_logs/resnet50_$(date +%Y%m%d_%H%M%S).log
