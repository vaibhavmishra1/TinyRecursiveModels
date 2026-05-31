from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nqweens.build_dataset import NQueensDatasetConfig, build_dataset
from scripts.train_gram_common import GRAMTrainConfig, train


DEFAULT_DATA_PATH = "nqweens/data/nqueens-8x8"
DEFAULT_CHECKPOINT_PATH = "checkpoints/GRAM-NQueens-8x8/gram_nqueens_8x8"


def make_config(args: argparse.Namespace | None = None) -> GRAMTrainConfig:
    data_path = getattr(args, "data_path", DEFAULT_DATA_PATH)
    checkpoint_path = getattr(args, "checkpoint_path", DEFAULT_CHECKPOINT_PATH)
    epochs = getattr(args, "epochs", 3000)
    eval_interval = getattr(args, "eval_interval", 300)
    global_batch_size = getattr(args, "global_batch_size", 768)
    device = getattr(args, "device", "auto")
    seed = getattr(args, "seed", 0)
    num_workers = getattr(args, "num_workers", 1)
    log_every = getattr(args, "log_every", 20)
    compile_model = getattr(args, "compile", False)
    load_checkpoint = getattr(args, "load_checkpoint", None)

    return GRAMTrainConfig(
        run_name="gram_nqueens_8x8",
        data_paths=[data_path],
        checkpoint_path=checkpoint_path,
        epochs=epochs,
        eval_interval=eval_interval,
        global_batch_size=global_batch_size,
        lr=1e-4,
        puzzle_emb_lr=1e-4,
        weight_decay=1.0,
        puzzle_emb_weight_decay=0.1,
        grad_clip=1.0,
        ema=True,
        ema_rate=0.9999,
        hidden_size=512,
        num_heads=8,
        expansion=1,
        H_layers=2,
        L_layers=2,
        T_steps=3,
        K_steps=4,
        N_sup=16,
        puzzle_emb_ndim=0,
        puzzle_emb_len=16,
        pos_encodings="rope",
        forward_dtype="bfloat16",
        mlp_t=False,
        decoder_swiglu=True,
        beta=0.07,
        kl_balance=0.8,
        act_loss_weight=1.0,
        lprm_loss_weight=1.0,
        loss_type="stablemax_cross_entropy",
        seed=seed,
        device=device,
        num_workers=num_workers,
        log_every=log_every,
        save_every_eval=True,
        compile=compile_model,
        load_checkpoint=load_checkpoint,
    )


def _dataset_exists(data_path: str) -> bool:
    root = Path(data_path)
    return (root / "train" / "dataset.json").exists() and (root / "test" / "dataset.json").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRAM on 8x8 N-Queens.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--global-batch-size", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--no-build-if-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_data or (not args.no_build_if_missing and not _dataset_exists(args.data_path)):
        build_dataset(NQueensDatasetConfig(output_dir=args.data_path, seed=args.seed))
    train(make_config(args))


if __name__ == "__main__":
    main()
