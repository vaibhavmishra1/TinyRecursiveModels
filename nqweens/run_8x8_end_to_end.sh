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
EMA="${EMA:-1}"
EMA_RATE="${EMA_RATE:-0.9999}"
NUM_INFER_PUZZLES="${NUM_INFER_PUZZLES:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
INFER_SPLIT="${INFER_SPLIT:-both}"
INFER_EVERY_EVAL="${INFER_EVERY_EVAL:-1}"
INFER_DISABLE_ACT="${INFER_DISABLE_ACT:-1}"
INFER_RAW_CHECKPOINT="${INFER_RAW_CHECKPOINT:-0}"
MIN_LOG_STD="${MIN_LOG_STD:--10.0}"
MAX_LOG_STD="${MAX_LOG_STD:-0.0}"
LPRM_DETACH_CORE="${LPRM_DETACH_CORE:-1}"
TRAIN_PRIOR_CARRY="${TRAIN_PRIOR_CARRY:-1}"
POSTERIOR_FINAL_ONLY="${POSTERIOR_FINAL_ONLY:-1}"

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
  --ema-rate "$EMA_RATE" \
  --no-build-if-missing \
  --infer-num-puzzles "$NUM_INFER_PUZZLES" \
  --infer-num-samples "$NUM_SAMPLES" \
  --infer-split "$INFER_SPLIT" \
  --min-log-std "$MIN_LOG_STD" \
  --max-log-std "$MAX_LOG_STD"
)
if [[ "$EMA" != "1" ]]; then
  train_args+=(--disable-ema)
fi
if [[ "$INFER_RAW_CHECKPOINT" == "1" ]]; then
  train_args+=(--infer-raw-checkpoint)
fi
if [[ "$LPRM_DETACH_CORE" == "1" ]]; then
  train_args+=(--lprm-detach-core)
else
  train_args+=(--lprm-train-core)
fi
if [[ "$TRAIN_PRIOR_CARRY" == "1" ]]; then
  train_args+=(--prior-carry)
else
  train_args+=(--posterior-carry)
fi
if [[ "$POSTERIOR_FINAL_ONLY" == "1" ]]; then
  train_args+=(--posterior-final-transition-only)
else
  train_args+=(--posterior-all-transitions)
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

INFER_CHECKPOINT="$CKPT_DIR"
if [[ "$INFER_RAW_CHECKPOINT" == "1" ]]; then
  INFER_CHECKPOINT="$("$PYTHON" - "$CKPT_DIR" <<'PY'
from pathlib import Path
import sys

checkpoint_dir = Path(sys.argv[1])
candidates = [p for p in checkpoint_dir.glob("step_*_raw") if p.is_file()]
if not candidates:
    raise SystemExit(f"No raw step_*_raw checkpoints found in {checkpoint_dir}")
print(max(candidates, key=lambda p: int(p.name.split("_")[1])))
PY
)"
fi

if [[ "$INFER_SPLIT" == "both" ]]; then
  final_infer_splits=(train test)
else
  final_infer_splits=("$INFER_SPLIT")
fi
for split in "${final_infer_splits[@]}"; do
  infer_args=(
    nqweens/infer_8x8.py
    --checkpoint "$INFER_CHECKPOINT"
    --data-path "$DATA_DIR"
    --split "$split"
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
done
