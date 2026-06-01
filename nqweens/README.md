# GRAM 8x8 N-Queens

This folder contains an end-to-end 8x8 N-Queens setup for the GRAM implementation on the `feature/gram-model` branch.

## Paper Settings

The dataset follows Appendix C.2.1 of the GRAM paper:

- Generate all complete 8x8 N-Queens solutions.
- Build partial inputs by removing `k in {5, 6, 7}` queens from each full board.
- Split 85:15 by unique input configuration to avoid input leakage.
- Flatten each board row-major to length 64.
- Vocabulary: `0=pad`, `1=empty`, `2=queen`.
- No learned puzzle embeddings; the model still prepends the paper's 16 zero-padded puzzle-token positions.

The training script uses the paper hyperparameters for 8x8 N-Queens:

- `D=512`, `Dh=512`, `heads=8`, `fL/fH=2` attention plus SwiGLU layers.
- `K=4` low-level refinements and `T=3` high-level stochastic transitions.
- `N_sup=16`, `beta=0.07`, KL balance `0.8`.
- AdamW `lr=1e-4`, weight decay `1.0`, gradient clip `1.0`.
- Paper global batch size `768`, EMA `0.9999`, epochs `3000`.
- For 4x RTX 4090 24 GB, the launch scripts default to `GLOBAL_BATCH_SIZE=256`, giving a per-GPU local batch size of `64`. `384` maps to local batch `96`, but it can OOM on 24 GB cards with this implementation.

## Commands

On the local machine, use the `brahma` conda environment:

```bash
conda activate brahma
cd /Users/vaibhav/Desktop/research/recursive_reasoning/TinyRecursiveModels
```

Build the dataset:

```bash
python nqweens/build_dataset.py --output-dir nqweens/data/nqueens-8x8
```

Train on 4x RTX 4090:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 4 nqweens/train_8x8.py \
  --data-path nqweens/data/nqueens-8x8 \
  --checkpoint-path checkpoints/GRAM-NQueens-8x8/gram_nqueens_8x8 \
  --global-batch-size 256 \
  --device cuda
```

By default, training saves an EMA checkpoint and runs N-Queens inference after every `--eval-interval` epochs. With the default `--eval-interval 300`, this produces `nqweens_eval_metrics.jsonl` in the checkpoint directory every 300 epochs. Add `--no-infer-every-eval` to disable periodic inference.

`torch.compile` is opt-in with `COMPILE=1` or `--compile`. On the tested PyTorch stack it can fail inside Inductor with a recursion error, so the run scripts default to eager mode. For `GLOBAL_BATCH_SIZE=256`, a 2-3 hour full run requires about `8-12` optimizer steps/sec.

Run inference with ACT halting and LPRM-ranked candidate selection, and report validity plus coverage with 20 samples:

```bash
python nqweens/infer_8x8.py \
  --checkpoint checkpoints/GRAM-NQueens-8x8/gram_nqueens_8x8 \
  --data-path nqweens/data/nqueens-8x8 \
  --split test \
  --num-puzzles 100 \
  --num-samples 20
```

To force fixed-depth 16-step sampling instead of ACT halting, add `--disable-act`.

Or run everything:

```bash
PYTHON=/Users/vaibhav/miniconda3/envs/brahma/bin/python NUM_GPUS=4 GLOBAL_BATCH_SIZE=256 bash nqweens/run_8x8_end_to_end.sh
```

For a quick smoke run, lower the epoch count:

```bash
PYTHON=/Users/vaibhav/miniconda3/envs/brahma/bin/python NUM_GPUS=1 EPOCHS=1 EVAL_INTERVAL=1 GLOBAL_BATCH_SIZE=64 NUM_INFER_PUZZLES=2 bash nqweens/run_8x8_end_to_end.sh
```

## VM One-Shot Run

On a CUDA VM, this script creates `.venv-nqweens`, installs PyTorch and the N-Queens dependencies, builds the dataset, trains, then runs inference:

```bash
NUM_GPUS=4 GLOBAL_BATCH_SIZE=256 bash nqweens/install_train_infer_8x8.sh
```

The script defaults to the PyTorch CUDA 12.6 wheel index. Override it if your VM image needs a different CUDA wheel:

```bash
NUM_GPUS=4 GLOBAL_BATCH_SIZE=256 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash nqweens/install_train_infer_8x8.sh
```

Useful VM smoke run:

```bash
NUM_GPUS=1 EPOCHS=1 EVAL_INTERVAL=1 GLOBAL_BATCH_SIZE=64 NUM_INFER_PUZZLES=2 bash nqweens/install_train_infer_8x8.sh
```

Disable periodic inference during training:

```bash
INFER_EVERY_EVAL=0 bash nqweens/install_train_infer_8x8.sh
```

Install dependencies only:

```bash
INSTALL_ONLY=1 bash nqweens/install_train_infer_8x8.sh
```
