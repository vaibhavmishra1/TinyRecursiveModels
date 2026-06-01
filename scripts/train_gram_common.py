from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional
import json
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from models.ema import EMAHelper
from models.losses import GRAMLossHead
from models.recursive_reasoning.gram import GenerativeRecursiveReasoningModel_ACTV1
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


@dataclass
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


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
    expansion: float = 1
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
    min_log_std: float = -10.0
    max_log_std: float = 2.0
    detach_lprm_core: bool = False

    beta: float = 0.1
    kl_balance: float = 0.8
    act_loss_weight: float = 1.0
    lprm_loss_weight: float = 1.0
    lprm_reward_type: str = "token_accuracy"
    prior_lprm_loss_weight: float = 0.0
    prior_aux_loss_weight: float = 0.0
    loss_type: str = "stablemax_cross_entropy"

    seed: int = 0
    device: str = "auto"
    num_workers: int = 1
    log_every: int = 20
    save_every_eval: bool = True
    compile: bool = False
    load_checkpoint: Optional[str] = None


def init_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(local_rank=int(os.environ.get("LOCAL_RANK", "0")))

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed GRAM training uses NCCL and requires CUDA.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        except TypeError:
            dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank, enabled=True)


def finalize_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        try:
            dist.barrier(device_ids=[context.local_rank])
        except TypeError:
            dist.barrier()


def select_device(name: str, context: Optional[DistributedContext] = None) -> torch.device:
    if context is not None and context.enabled:
        return torch.device("cuda", context.local_rank)
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_loader(config: GRAMTrainConfig, split: str, epochs_per_iter: int, rank: int = 0, world_size: int = 1):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths,
            global_batch_size=config.global_batch_size,
            test_set_mode=(split == "test"),
            epochs_per_iter=epochs_per_iter,
            rank=rank,
            num_replicas=world_size,
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


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix("_orig_mod.").replace("._orig_mod.", "."): v for k, v in state_dict.items()}


def _unwrap_compiled(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _state_dict_for_save(model: nn.Module) -> dict[str, torch.Tensor]:
    return _strip_compile_prefix(_unwrap_compiled(model).state_dict())


def _broadcast_model_state(model: nn.Module) -> None:
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor, src=0)


def _compile_for_training(loss_model: GRAMLossHead) -> nn.Module:
    # Compiling the whole GRAMLossHead asks Dynamo/Inductor to trace the full
    # N_sup supervision loop. Compiling the inner recurrent model keeps first
    # compile time much lower while still optimizing the repeated heavy block.
    loss_model.model.inner = torch.compile(loss_model.model.inner, mode="reduce-overhead")  # type: ignore[assignment]
    return loss_model


def build_model(config: GRAMTrainConfig, metadata, device: torch.device, rank: int = 0, world_size: int = 1):
    if config.global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size={config.global_batch_size} must be divisible by world_size={world_size}")

    model_cfg = {
        "batch_size": config.global_batch_size // world_size,
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
        "min_log_std": config.min_log_std,
        "max_log_std": config.max_log_std,
        "detach_lprm_core": config.detach_lprm_core,
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
        lprm_reward_type=config.lprm_reward_type,
        prior_lprm_loss_weight=config.prior_lprm_loss_weight,
        prior_aux_loss_weight=config.prior_aux_loss_weight,
    )
    loss_model.to(device)

    if config.load_checkpoint and rank == 0:
        state = torch.load(config.load_checkpoint, map_location=device)
        state = _strip_compile_prefix(state)
        loss_model.load_state_dict(state, strict=False)

    if world_size > 1:
        _broadcast_model_state(loss_model)

    if config.compile and device.type == "cuda":
        loss_model = _compile_for_training(loss_model)  # type: ignore[assignment]
    return loss_model


def create_optimizers(config: GRAMTrainConfig, model: nn.Module, world_size: int = 1):
    dense_params = [p for p in model.parameters() if p.requires_grad]
    optimizers = [torch.optim.AdamW(dense_params, lr=config.lr, weight_decay=config.weight_decay)]

    # ARC puzzle embeddings are intentionally sparse and very large. The existing
    # RRM codebase stores them as buffers and updates only touched rows.
    model_for_attrs = _unwrap_compiled(model)
    puzzle_emb = getattr(getattr(model_for_attrs, "model", model_for_attrs), "puzzle_emb", None)
    if config.puzzle_emb_ndim > 0 and puzzle_emb is not None:
        optimizers.insert(
            0,
            CastedSparseEmbeddingSignSGD_Distributed(
                puzzle_emb.buffers(),
                lr=config.puzzle_emb_lr,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
        )
    return optimizers, dense_params


def save_checkpoint(path: Path, model: nn.Module, step: int, ema_helper: Optional[EMAHelper] = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = path / f"step_{step}"
    if ema_helper is not None:
        torch.save(_state_dict_for_save(model), path / f"step_{step}_raw")
    model_to_save = ema_helper.ema_copy(model) if ema_helper is not None else model
    torch.save(_state_dict_for_save(model_to_save), checkpoint_file)
    if model_to_save is not model:
        del model_to_save
    return checkpoint_file


def _all_reduce_gradients(params: List[torch.Tensor], context: DistributedContext) -> None:
    if not context.enabled:
        return
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)


def _reduce_metrics(metrics: dict[str, torch.Tensor], context: DistributedContext) -> Optional[dict[str, float]]:
    if not metrics:
        return {}

    metric_keys = sorted(metrics.keys())
    metric_values = torch.stack([metrics[k].detach().to(torch.float32) for k in metric_keys])
    if context.enabled:
        dist.reduce(metric_values, dst=0, op=dist.ReduceOp.SUM)
    if not context.is_main:
        return None
    return {k: float(metric_values[i].cpu()) for i, k in enumerate(metric_keys)}


def _reduce_scalar(value: torch.Tensor, context: DistributedContext) -> Optional[float]:
    value = value.detach().to(torch.float32)
    if context.enabled:
        dist.reduce(value, dst=0, op=dist.ReduceOp.SUM)
    if not context.is_main:
        return None
    return float(value.cpu())


def train(
    config: GRAMTrainConfig,
    on_checkpoint: Optional[Callable[[int, int, Path], None]] = None,
):
    context = init_distributed()
    try:
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(config.seed + context.rank)
        device = select_device(config.device, context)
        checkpoint_path = Path(config.checkpoint_path)
        if context.is_main:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            with (checkpoint_path / "gram_train_config.json").open("w") as f:
                json.dump(asdict(config), f, indent=2)
        barrier(context)

        train_epochs_per_iter = config.eval_interval if config.eval_interval else config.epochs
        total_iters = config.epochs // train_epochs_per_iter
        if config.epochs % train_epochs_per_iter != 0:
            raise ValueError("eval_interval must divide epochs")

        train_loader, train_metadata = create_loader(
            config,
            "train",
            train_epochs_per_iter,
            rank=context.rank,
            world_size=context.world_size,
        )
        model = build_model(config, train_metadata, device, rank=context.rank, world_size=context.world_size)
        optimizers, dense_params = create_optimizers(config, model, world_size=context.world_size)

        ema_helper = None
        if config.ema:
            ema_helper = EMAHelper(mu=config.ema_rate)
            ema_helper.register(model)

        total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
        step = 0
        log_path = checkpoint_path / "train_metrics.jsonl"
        started = time.time()

        if context.is_main:
            local_batch_size = config.global_batch_size // context.world_size
            print(f"device={device}")
            print(f"world_size={context.world_size}")
            print(f"global_batch_size={config.global_batch_size}")
            print(f"local_batch_size={local_batch_size}")
            print(f"compile={config.compile}")
            if config.compile:
                print("compile_mode=inner_reduce_overhead")
            print(f"total_steps={total_steps}")
            print(f"metadata={train_metadata}")

        for iter_id in range(total_iters):
            if context.is_main:
                print(f"epoch={iter_id * train_epochs_per_iter}")
            model.train()
            carry = None

            for _set_name, batch, global_batch_size in train_loader:
                step += 1
                if step > total_steps:
                    break

                batch = {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in batch.items()}
                if carry is None:
                    with torch.device(device):
                        carry = model.initial_carry(batch)  # type: ignore[attr-defined]

                carry, loss, metrics, _outputs, _finished = model(carry=carry, batch=batch, return_keys=[])
                (loss / global_batch_size).backward()

                _all_reduce_gradients(dense_params, context)

                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(dense_params, config.grad_clip)

                for optim in optimizers:
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                if ema_helper is not None:
                    ema_helper.update(model)

                if step % config.log_every == 0:
                    reduced = _reduce_metrics(metrics, context)
                    reduced_loss = _reduce_scalar(loss, context)
                    if context.is_main and reduced is not None and reduced_loss is not None:
                        elapsed = time.time() - started
                        count = max(reduced.pop("count", 1.0), 1.0)
                        normalized = {}
                        for k, v in reduced.items():
                            if k.endswith("loss"):
                                normalized[k] = v / global_batch_size
                            elif k in {"accuracy", "exact_accuracy", "q_halt_accuracy", "steps"}:
                                normalized[k] = v / count
                            elif k in {
                                "lprm_reward",
                                "prior_lprm_reward",
                                "prior_exact_accuracy",
                                "prior_queen_count",
                                "prior_conflicts",
                                "prior_clue_violations",
                            }:
                                normalized[k] = v / count
                            elif k in {
                                "prior_std",
                                "sample_std",
                                "prior_log_std_mean",
                                "prior_log_std_max",
                                "sample_log_std_mean",
                                "sample_log_std_max",
                            }:
                                normalized[k] = v / context.world_size
                            else:
                                normalized[k] = v
                        normalized.update({"step": step, "elapsed_seconds": elapsed, "loss": reduced_loss / global_batch_size})
                        print(normalized, flush=True)
                        with log_path.open("a") as f:
                            f.write(json.dumps(normalized) + "\n")

            if config.save_every_eval:
                if context.is_main:
                    checkpoint_file = save_checkpoint(checkpoint_path, model, step, ema_helper)
                    if on_checkpoint is not None:
                        on_checkpoint(iter_id * train_epochs_per_iter + train_epochs_per_iter, step, checkpoint_file)
                barrier(context)

        if context.is_main:
            save_checkpoint(checkpoint_path, model, step, ema_helper)
        barrier(context)
    finally:
        finalize_distributed(context)
