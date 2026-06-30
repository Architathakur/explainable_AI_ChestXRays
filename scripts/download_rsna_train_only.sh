#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/raw/rsna}"
ARCHIVE="$DATA_DIR/rsna-pneumonia-detection-challenge.zip"

mkdir -p "$DATA_DIR"

.venv-train/bin/kaggle competitions download \
  -c rsna-pneumonia-detection-challenge \
  -p "$DATA_DIR" \
  --force

unzip -n "$ARCHIVE" \
  stage_2_train_labels.csv \
  'stage_2_train_images/*' \
  -d "$DATA_DIR"

rm -f "$ARCHIVE"

du -sh "$DATA_DIR"
