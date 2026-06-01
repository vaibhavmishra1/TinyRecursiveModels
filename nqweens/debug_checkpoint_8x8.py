from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nqweens.common import conflicts, is_valid_solution
from nqweens.infer_8x8 import _load_metadata, _load_train_config, _strip_state_prefixes
from scripts.train_gram_common import build_model, select_device


def _load_examples(data_path: str, split: str, count: int) -> dict[str, np.ndarray]:
    root = Path(data_path) / split
    inputs = np.load(root / "all__inputs.npy", mmap_mode="r")[:count].astype(np.int32)
    labels = np.load(root / "all__labels.npy", mmap_mode="r")[:count].astype(np.int32)
    puzzle_indices = np.load(root / "all__puzzle_indices.npy")
    puzzle_identifiers = np.load(root / "all__puzzle_identifiers.npy")

    example_to_puzzle = np.searchsorted(puzzle_indices, np.arange(count), side="right") - 1
    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers[example_to_puzzle].astype(np.int32),
    }


def _to_torch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def _stats(preds: np.ndarray, labels: np.ndarray, inputs: np.ndarray, n: int) -> dict[str, float]:
    exact = np.all(preds == labels, axis=1)
    queen_counts = (preds == 2).sum(axis=1)
    conflict_counts = np.asarray([conflicts(pred, n=n) for pred in preds], dtype=np.float32)
    clue_violations = ((inputs == 2) & (preds != 2)).sum(axis=1)
    valid = np.asarray([is_valid_solution(pred, n=n, clues=inp) for pred, inp in zip(preds, inputs)], dtype=np.float32)
    return {
        "exact_accuracy": float(np.mean(exact)),
        "valid_accuracy": float(np.mean(valid)),
        "queen_count_mean": float(np.mean(queen_counts)),
        "queen_count_std": float(np.std(queen_counts)),
        "conflicts_mean": float(np.mean(conflict_counts)),
        "clue_violations_mean": float(np.mean(clue_violations)),
    }


def _fresh_prior_rollout(model: torch.nn.Module, batch: dict[str, torch.Tensor], steps: int) -> np.ndarray:
    inner = model.inner
    with torch.device(batch["inputs"].device):
        carry = inner.empty_carry(batch["inputs"].shape[0])
    carry = inner.reset_carry(torch.ones((batch["inputs"].shape[0],), dtype=torch.bool, device=batch["inputs"].device), carry)
    outputs = None
    model.eval()
    with torch.inference_mode():
        for _ in range(steps):
            carry, outputs = inner(carry, batch, force_prior=True)
    if outputs is None:
        raise RuntimeError("Fresh prior rollout produced no outputs")
    return outputs["logits"].argmax(dim=-1).detach().cpu().numpy().astype(np.int32)


def _posterior_contaminated_prior(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    posterior_steps: int,
) -> np.ndarray:
    model.train()
    with torch.device(batch["inputs"].device):
        carry = model.initial_carry(batch)
    with torch.no_grad():
        for _ in range(posterior_steps):
            carry, _outputs = model(carry=carry, batch=batch, force_prior=False)
        _carry, outputs = model(carry=carry, batch=batch, force_prior=True)
    return outputs["logits"].argmax(dim=-1).detach().cpu().numpy().astype(np.int32)


def run_debug(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint)
    device = select_device(args.device)
    metadata = _load_metadata(args.data_path, args.split)
    train_config = _load_train_config(checkpoint_path, args.data_path, str(checkpoint_path.parent), args.device)
    train_config.global_batch_size = args.batch_size
    train_config.device = args.device

    loss_model = build_model(train_config, metadata, device)
    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = loss_model.load_state_dict(_strip_state_prefixes(state), strict=False)
    if missing:
        print(f"Missing checkpoint keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    np_batch = _load_examples(args.data_path, args.split, args.batch_size)
    torch_batch = _to_torch(np_batch, device)
    model = loss_model.model

    fresh_preds = _fresh_prior_rollout(model, torch_batch, args.steps)
    contaminated_preds = _posterior_contaminated_prior(model, torch_batch, args.posterior_steps)

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "batch_size": args.batch_size,
        "fresh_prior_steps": args.steps,
        "posterior_contamination_steps": args.posterior_steps,
        "fresh_prior": _stats(fresh_preds, np_batch["labels"], np_batch["inputs"], args.n),
        "posterior_contaminated_prior": _stats(contaminated_preds, np_batch["labels"], np_batch["inputs"], args.n),
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fresh prior rollout with posterior-contaminated carry prior on N-Queens.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="nqweens/data/nqueens-8x8")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--posterior-steps", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run_debug(parse_args())
