#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-nqweens/data/nqueens-8x8}"
CKPT_DIR="${CKPT_DIR:-checkpoints/GRAM-NQueens-8x8/gram_nqueens_8x8}"
EPOCHS="${EPOCHS:-3000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-300}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-0}"
NUM_INFER_PUZZLES="${NUM_INFER_PUZZLES:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
INFER_EVERY_EVAL="${INFER_EVERY_EVAL:-1}"
INFER_DISABLE_ACT="${INFER_DISABLE_ACT:-1}"
MIN_LOG_STD="${MIN_LOG_STD:--10.0}"
MAX_LOG_STD="${MAX_LOG_STD:-0.0}"
LPRM_DETACH_CORE="${LPRM_DETACH_CORE:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "$COMPILE" == "1" ]]; then
  export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
fi

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
  --infer-num-samples "$NUM_SAMPLES" \
  --min-log-std "$MIN_LOG_STD" \
  --max-log-std "$MAX_LOG_STD"
)
if [[ "$LPRM_DETACH_CORE" == "1" ]]; then
  train_args+=(--lprm-detach-core)
else
  train_args+=(--lprm-train-core)
fi
if [[ "$INFER_EVERY_EVAL" != "1" ]]; then
  train_args+=(--no-infer-every-eval)
fi
if [[ "$INFER_DISABLE_ACT" == "1" ]]; then
  train_args+=(--infer-disable-act)
else
  train_args+=(--infer-use-act)
fi
if [[ "$COMPILE" == "1" ]]; then
  train_args+=(--compile)
fi
if [[ "$NUM_GPUS" -gt 1 ]]; then
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc-per-node "$NUM_GPUS" \
    "${train_args[@]}"
else
  "$PYTHON" "${train_args[@]}"
fi

infer_args=(
  nqweens/infer_8x8.py
  --checkpoint "$CKPT_DIR"
  --data-path "$DATA_DIR"
  --split test
  --num-puzzles "$NUM_INFER_PUZZLES"
  --num-samples "$NUM_SAMPLES"
  --steps 16
  --device "$DEVICE"
)
if [[ "$INFER_DISABLE_ACT" == "1" ]]; then
  infer_args+=(--disable-act)
else
  infer_args+=(--use-act)
fi
"$PYTHON" "${infer_args[@]}"
