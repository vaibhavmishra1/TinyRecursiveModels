from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nqweens.build_dataset import NQueensDatasetConfig, build_dataset
from scripts.train_gram_common import GRAMTrainConfig, train


DEFAULT_DATA_PATH = "nqweens/data/nqueens-8x8"
DEFAULT_CHECKPOINT_PATH = "checkpoints/GRAM-NQueens-8x8/gram_nqueens_8x8"


def make_config(args: argparse.Namespace | None = None) -> GRAMTrainConfig:
    def get(name: str, default):
        return getattr(args, name, default)

    data_path = get("data_path", DEFAULT_DATA_PATH)
    checkpoint_path = get("checkpoint_path", DEFAULT_CHECKPOINT_PATH)

    return GRAMTrainConfig(
        run_name=get("run_name", "gram_nqueens_8x8"),
        data_paths=[data_path],
        checkpoint_path=checkpoint_path,
        epochs=get("epochs", 3000),
        eval_interval=get("eval_interval", 300),
        global_batch_size=get("global_batch_size", 768),
        lr=get("lr", 1e-4),
        puzzle_emb_lr=get("puzzle_emb_lr", 1e-4),
        weight_decay=get("weight_decay", 1.0),
        puzzle_emb_weight_decay=get("puzzle_emb_weight_decay", 0.1),
        grad_clip=get("grad_clip", 1.0),
        ema=get("ema", True),
        ema_rate=get("ema_rate", 0.9999),
        hidden_size=get("hidden_size", 512),
        num_heads=get("num_heads", 8),
        expansion=get("expansion", 1),
        H_layers=get("H_layers", 2),
        L_layers=get("L_layers", 2),
        T_steps=get("T_steps", 3),
        K_steps=get("K_steps", 4),
        N_sup=get("N_sup", 16),
        puzzle_emb_ndim=get("puzzle_emb_ndim", 0),
        puzzle_emb_len=get("puzzle_emb_len", 16),
        pos_encodings=get("pos_encodings", "rope"),
        forward_dtype=get("forward_dtype", "bfloat16"),
        mlp_t=get("mlp_t", False),
        decoder_swiglu=get("decoder_swiglu", True),
        beta=get("beta", 0.07),
        kl_balance=get("kl_balance", 0.8),
        act_loss_weight=get("act_loss_weight", 1.0),
        lprm_loss_weight=get("lprm_loss_weight", 1.0),
        loss_type=get("loss_type", "stablemax_cross_entropy"),
        seed=get("seed", 0),
        device=get("device", "auto"),
        num_workers=get("num_workers", 1),
        log_every=get("log_every", 20),
        save_every_eval=get("save_every_eval", True),
        compile=get("compile", False),
        load_checkpoint=get("load_checkpoint", None),
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
    parser.add_argument("--no-infer-every-eval", action="store_true")
    parser.add_argument("--infer-split", choices=["train", "test"], default="test")
    parser.add_argument("--infer-num-puzzles", type=int, default=100)
    parser.add_argument("--infer-num-samples", type=int, default=20)
    parser.add_argument("--infer-steps", type=int, default=16)
    parser.add_argument("--infer-print-puzzles", type=int, default=0)
    parser.add_argument("--infer-print-samples", type=int, default=0)
    parser.add_argument("--infer-disable-act", action="store_true")
    return parser.parse_args()


def make_inference_callback(args: argparse.Namespace):
    if args.no_infer_every_eval:
        return None

    def _callback(epoch: int, step: int, checkpoint_file: Path) -> None:
        from nqweens.infer_8x8 import run_inference

        print(f"INFER epoch={epoch} step={step} checkpoint={checkpoint_file}", flush=True)
        summary = run_inference(
            argparse.Namespace(
                checkpoint=str(checkpoint_file),
                data_path=args.data_path,
                split=args.infer_split,
                n=8,
                num_puzzles=args.infer_num_puzzles,
                num_samples=args.infer_num_samples,
                steps=args.infer_steps,
                device=args.device,
                disable_act=args.infer_disable_act,
                print_puzzles=args.infer_print_puzzles,
                print_samples=args.infer_print_samples,
            )
        )
        summary.update({"epoch": epoch, "train_step": step})
        metrics_path = Path(args.checkpoint_path) / "nqweens_eval_metrics.jsonl"
        with metrics_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return _callback


def main() -> None:
    args = parse_args()
    if args.rebuild_data or (not args.no_build_if_missing and not _dataset_exists(args.data_path)):
        build_dataset(NQueensDatasetConfig(output_dir=args.data_path, seed=args.seed))
    train(make_config(args), on_checkpoint=make_inference_callback(args))


if __name__ == "__main__":
    main()
