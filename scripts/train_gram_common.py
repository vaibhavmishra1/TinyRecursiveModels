from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.ema import EMAHelper
from models.losses import GRAMLossHead
from models.recursive_reasoning.gram import GenerativeRecursiveReasoningModel_ACTV1
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


@dataclass
class GRAMTrainConfig:
    run_name: str
    data_paths: List[str]
    checkpoint_path: str

    epochs: int
    eval_interval: int
    global_batch_size: int = 768

    lr: float = 1e-4
    puzzle_emb_lr: float = 1e-4
    weight_decay: float = 1.0
    puzzle_emb_weight_decay: float = 0.1
    grad_clip: float = 1.0

    ema: bool = True
    ema_rate: float = 0.9999

    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4
    H_layers: int = 2
    L_layers: int = 2
    T_steps: int = 3
    K_steps: int = 4
    N_sup: int = 16

    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16
    pos_encodings: str = "rope"
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    decoder_swiglu: bool = True

    beta: float = 0.1
    kl_balance: float = 0.8
    act_loss_weight: float = 1.0
    lprm_loss_weight: float = 1.0
    loss_type: str = "stablemax_cross_entropy"

    seed: int = 0
    device: str = "auto"
    num_workers: int = 1
    log_every: int = 20
    save_every_eval: bool = True
    compile: bool = False
    load_checkpoint: Optional[str] = None


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_loader(config: GRAMTrainConfig, split: str, epochs_per_iter: int):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths,
            global_batch_size=config.global_batch_size,
            test_set_mode=(split == "test"),
            epochs_per_iter=epochs_per_iter,
            rank=0,
            num_replicas=1,
        ),
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_workers,
        prefetch_factor=8 if config.num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    return loader, dataset.metadata


def build_model(config: GRAMTrainConfig, metadata, device: torch.device):
    model_cfg = {
        "batch_size": config.global_batch_size,
        "seq_len": metadata.seq_len,
        "puzzle_emb_ndim": config.puzzle_emb_ndim,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
        "vocab_size": metadata.vocab_size,
        "K_steps": config.K_steps,
        "T_steps": config.T_steps,
        "N_sup": config.N_sup,
        "H_layers": config.H_layers,
        "L_layers": config.L_layers,
        "hidden_size": config.hidden_size,
        "expansion": config.expansion,
        "num_heads": config.num_heads,
        "pos_encodings": config.pos_encodings,
        "forward_dtype": config.forward_dtype,
        "halt_max_steps": config.N_sup,
        "halt_exploration_prob": 0.0,
        "puzzle_emb_len": config.puzzle_emb_len,
        "mlp_t": config.mlp_t,
        "decoder_swiglu": config.decoder_swiglu,
    }
    model = GenerativeRecursiveReasoningModel_ACTV1(model_cfg)
    loss_model = GRAMLossHead(
        model,
        loss_type=config.loss_type,
        deep_supervision_steps=config.N_sup,
        beta=config.beta,
        kl_balance=config.kl_balance,
        act_loss_weight=config.act_loss_weight,
        lprm_loss_weight=config.lprm_loss_weight,
    )
    loss_model.to(device)

    if config.load_checkpoint:
        state = torch.load(config.load_checkpoint, map_location=device)
        loss_model.load_state_dict(state, strict=False)

    if config.compile and device.type == "cuda":
        loss_model = torch.compile(loss_model)  # type: ignore[assignment]
    return loss_model


def create_optimizers(config: GRAMTrainConfig, model: nn.Module):
    dense_params = [p for p in model.parameters() if p.requires_grad]
    optimizers = [torch.optim.AdamW(dense_params, lr=config.lr, weight_decay=config.weight_decay)]

    # ARC puzzle embeddings are intentionally sparse and very large. The existing
    # RRM codebase stores them as buffers and updates only touched rows.
    puzzle_emb = getattr(getattr(model, "model", model), "puzzle_emb", None)
    if config.puzzle_emb_ndim > 0 and puzzle_emb is not None:
        optimizers.insert(
            0,
            CastedSparseEmbeddingSignSGD_Distributed(
                puzzle_emb.buffers(),
                lr=config.puzzle_emb_lr,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=1,
            ),
        )
    return optimizers, dense_params


def save_checkpoint(path: Path, model: nn.Module, step: int):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / f"step_{step}")


def train(config: GRAMTrainConfig):
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    with (checkpoint_path / "gram_train_config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    train_epochs_per_iter = config.eval_interval if config.eval_interval else config.epochs
    total_iters = config.epochs // train_epochs_per_iter
    if config.epochs % train_epochs_per_iter != 0:
        raise ValueError("eval_interval must divide epochs")

    train_loader, train_metadata = create_loader(config, "train", train_epochs_per_iter)
    model = build_model(config, train_metadata, device)
    optimizers, dense_params = create_optimizers(config, model)

    ema_helper = None
    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(model)

    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    step = 0
    log_path = checkpoint_path / "train_metrics.jsonl"
    started = time.time()

    print(f"device={device}")
    print(f"total_steps={total_steps}")
    print(f"metadata={train_metadata}")

    for iter_id in range(total_iters):
        print(f"epoch={iter_id * train_epochs_per_iter}")
        model.train()
        carry = None

        for _set_name, batch, global_batch_size in train_loader:
            step += 1
            if step > total_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            if carry is None:
                with torch.device(device):
                    carry = model.initial_carry(batch)  # type: ignore[attr-defined]

            carry, loss, metrics, _outputs, _finished = model(carry=carry, batch=batch, return_keys=[])
            (loss / global_batch_size).backward()

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(dense_params, config.grad_clip)

            for optim in optimizers:
                optim.step()
                optim.zero_grad(set_to_none=True)

            if ema_helper is not None:
                ema_helper.update(model)

            if step % config.log_every == 0:
                elapsed = time.time() - started
                reduced = {k: float(v.detach().cpu()) for k, v in metrics.items()}
                count = max(reduced.pop("count", 1.0), 1.0)
                normalized = {}
                for k, v in reduced.items():
                    if k.endswith("loss"):
                        normalized[k] = v / global_batch_size
                    elif k in {"accuracy", "exact_accuracy", "q_halt_accuracy", "steps"}:
                        normalized[k] = v / count
                    else:
                        normalized[k] = v
                normalized.update({"step": step, "elapsed_seconds": elapsed, "loss": float(loss.detach().cpu()) / global_batch_size})
                print(normalized, flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps(normalized) + "\n")

        if config.save_every_eval:
            save_checkpoint(checkpoint_path, model, step)

    save_checkpoint(checkpoint_path, model, step)
