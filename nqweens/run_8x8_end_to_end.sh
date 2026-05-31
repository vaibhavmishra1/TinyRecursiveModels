#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-nqweens/data/nqueens-8x8}"
CKPT_DIR="${CKPT_DIR:-checkpoints/GRAM-NQueens-8x8/gram_nqueens_8x8}"
EPOCHS="${EPOCHS:-3000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-300}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
DEVICE="${DEVICE:-auto}"
NUM_INFER_PUZZLES="${NUM_INFER_PUZZLES:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
INFER_EVERY_EVAL="${INFER_EVERY_EVAL:-1}"

"$PYTHON" nqweens/build_dataset.py \
  --output-dir "$DATA_DIR" \
  --n 8 \
  --remove-counts 5,6,7 \
  --train-fraction 0.85 \
  --seed 0

train_args=(
  nqweens/train_8x8.py
  --data-path "$DATA_DIR" \
  --checkpoint-path "$CKPT_DIR" \
  --epochs "$EPOCHS" \
  --eval-interval "$EVAL_INTERVAL" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --device "$DEVICE" \
  --no-build-if-missing \
  --infer-num-puzzles "$NUM_INFER_PUZZLES" \
  --infer-num-samples "$NUM_SAMPLES"
)
if [[ "$INFER_EVERY_EVAL" != "1" ]]; then
  train_args+=(--no-infer-every-eval)
fi
"$PYTHON" "${train_args[@]}"

"$PYTHON" nqweens/infer_8x8.py \
  --checkpoint "$CKPT_DIR" \
  --data-path "$DATA_DIR" \
  --split test \
  --num-puzzles "$NUM_INFER_PUZZLES" \
  --num-samples "$NUM_SAMPLES" \
  --steps 16 \
  --device "$DEVICE"
