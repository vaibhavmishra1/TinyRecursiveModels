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
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
DEVICE="${DEVICE:-cuda}"
NUM_INFER_PUZZLES="${NUM_INFER_PUZZLES:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
INFER_EVERY_EVAL="${INFER_EVERY_EVAL:-1}"
RUN_INFER="${RUN_INFER:-1}"
INSTALL_ONLY="${INSTALL_ONLY:-0}"

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
    print(f"gpu={torch.cuda.get_device_name(0)}")
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
  --no-build-if-missing \
  --infer-num-puzzles "$NUM_INFER_PUZZLES" \
  --infer-num-samples "$NUM_SAMPLES"
)
if [[ "$INFER_EVERY_EVAL" != "1" ]]; then
  train_args+=(--no-infer-every-eval)
fi
python "${train_args[@]}"

if [[ "$RUN_INFER" == "1" ]]; then
  python nqweens/infer_8x8.py \
    --checkpoint "$CKPT_DIR" \
    --data-path "$DATA_DIR" \
    --split test \
    --num-puzzles "$NUM_INFER_PUZZLES" \
    --num-samples "$NUM_SAMPLES" \
    --steps 16 \
    --device "$DEVICE"
fi
