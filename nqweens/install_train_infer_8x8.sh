#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-nqweens}"
USE_VENV="${USE_VENV:-1}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

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
INFER_EVERY_EVAL="${INFER_EVERY_EVAL:-1}"
INFER_DISABLE_ACT="${INFER_DISABLE_ACT:-1}"
INFER_RAW_CHECKPOINT="${INFER_RAW_CHECKPOINT:-0}"
MIN_LOG_STD="${MIN_LOG_STD:--10.0}"
MAX_LOG_STD="${MAX_LOG_STD:-0.0}"
LPRM_DETACH_CORE="${LPRM_DETACH_CORE:-1}"
RUN_INFER="${RUN_INFER:-1}"
INSTALL_ONLY="${INSTALL_ONLY:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "$COMPILE" == "1" ]]; then
  export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
fi

if [[ "$USE_VENV" == "1" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

if [[ "$SKIP_INSTALL" != "1" ]]; then
  python -m pip install --upgrade pip wheel setuptools

  if [[ "$SKIP_TORCH_INSTALL" != "1" ]]; then
    python -m pip install --upgrade torch --index-url "$TORCH_INDEX_URL"
  fi

  python -m pip install -r nqweens/requirements.txt
fi

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_version={torch.version.cuda}")
    print(f"gpu_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"gpu[{i}]={torch.cuda.get_device_name(i)}")
PY

if [[ "$INSTALL_ONLY" == "1" ]]; then
  exit 0
fi

python nqweens/build_dataset.py \
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
  python -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc-per-node "$NUM_GPUS" \
    "${train_args[@]}"
else
  python "${train_args[@]}"
fi

if [[ "$RUN_INFER" == "1" ]]; then
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
  python "${infer_args[@]}"
fi
