from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.common import PuzzleDatasetMetadata
from nqweens.common import board_to_text, conflicts, is_valid_solution
from nqweens.train_8x8 import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATA_PATH, make_config
from scripts.train_gram_common import build_model, select_device


def _latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(
        (p for p in checkpoint_dir.glob("step_*") if p.is_file()),
        key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_", 1)[1].isdigit() else -1,
    )
    if not candidates:
        raise FileNotFoundError(f"No step_* checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def _strip_state_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _load_instances(data_path: str, split: str) -> List[dict]:
    path = Path(data_path) / split / "instances.json"
    with path.open("r") as f:
        return json.load(f)


def _load_metadata(data_path: str, split: str) -> PuzzleDatasetMetadata:
    with (Path(data_path) / split / "dataset.json").open("r") as f:
        return PuzzleDatasetMetadata(**json.load(f))


def _load_train_config(checkpoint: Path, data_path: str, checkpoint_path: str, device: str):
    config_path = checkpoint.parent / "gram_train_config.json"
    if config_path.exists():
        with config_path.open("r") as f:
            payload = json.load(f)
        payload["data_path"] = data_path
        payload["checkpoint_path"] = checkpoint_path
        payload["device"] = device
        payload["load_checkpoint"] = None
        return make_config(argparse.Namespace(**payload))

    args = argparse.Namespace(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        epochs=3000,
        eval_interval=300,
        global_batch_size=768,
        device=device,
        seed=0,
        num_workers=1,
        log_every=20,
        compile=False,
        load_checkpoint=None,
    )
    return make_config(args)


def _sample_model(
    model: torch.nn.Module,
    inputs: np.ndarray,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    batch = {
        "inputs": torch.from_numpy(inputs.astype(np.int32)).to(device),
        "puzzle_identifiers": torch.zeros((inputs.shape[0],), dtype=torch.int32, device=device),
    }

    inner = model.inner
    with torch.device(device):
        inner_carry = inner.empty_carry(inputs.shape[0])
    reset = torch.ones((inputs.shape[0],), dtype=torch.bool, device=device)
    inner_carry = inner.reset_carry(reset, inner_carry)

    outputs = None
    with torch.inference_mode():
        for _ in range(steps):
            inner_carry, outputs = inner(inner_carry, batch)
    if outputs is None:
        raise RuntimeError("Inference produced no outputs")
    return outputs["logits"].argmax(dim=-1).detach().cpu().numpy().astype(np.uint8)


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = _latest_checkpoint(checkpoint_path)

    data_path = args.data_path
    metadata = _load_metadata(data_path, args.split)
    device = select_device(args.device)
    train_config = _load_train_config(checkpoint_path, data_path, str(checkpoint_path.parent), args.device)
    train_config.global_batch_size = args.num_samples
    train_config.device = args.device

    loss_model = build_model(train_config, metadata, device)
    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = loss_model.load_state_dict(_strip_state_prefixes(state), strict=False)
    if unexpected:
        print(f"Unexpected checkpoint keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if missing:
        print(f"Missing checkpoint keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    base_model = loss_model.model
    base_model.eval()
    instances = _load_instances(data_path, args.split)[: args.num_puzzles]

    total_samples = 0
    valid_samples = 0
    first_sample_valid = 0
    coverage_values = []
    printed = 0

    for puzzle_index, instance in enumerate(instances):
        input_seq = np.asarray(instance["input"], dtype=np.uint8)
        inputs = np.repeat(input_seq[None, :], args.num_samples, axis=0)
        preds = _sample_model(base_model, inputs, args.steps, device)

        target_solutions = {tuple(int(x) for x in solution) for solution in instance["solutions"]}
        valid_solution_keys = []
        for pred in preds:
            pred_key = tuple(int(x) for x in pred)
            if is_valid_solution(pred, n=args.n, clues=input_seq):
                valid_samples += 1
                if pred_key in target_solutions:
                    valid_solution_keys.append(pred_key)
        if is_valid_solution(preds[0], n=args.n, clues=input_seq):
            first_sample_valid += 1

        unique_valid = set(valid_solution_keys)
        coverage = len(unique_valid) / max(len(target_solutions), 1)
        coverage_values.append(coverage)
        total_samples += len(preds)

        if printed < args.print_puzzles:
            printed += 1
            print(f"\nPuzzle {puzzle_index} input:")
            print(board_to_text(input_seq, n=args.n))
            print(f"targets={len(target_solutions)} unique_valid_samples={len(unique_valid)} coverage={coverage:.3f}")
            for sample_index, pred in enumerate(preds[: args.print_samples]):
                print(f"\nSample {sample_index} valid={is_valid_solution(pred, n=args.n, clues=input_seq)} conflicts={conflicts(pred, n=args.n)}")
                print(board_to_text(pred, n=args.n))

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_puzzles": len(instances),
        "num_samples_per_puzzle": args.num_samples,
        "steps": args.steps,
        "sample_accuracy": valid_samples / max(total_samples, 1),
        "single_sample_accuracy": first_sample_valid / max(len(instances), 1),
        "coverage_at_samples": float(np.mean(coverage_values)) if coverage_values else 0.0,
    }
    print("\n" + json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and evaluate a GRAM 8x8 N-Queens checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--num-puzzles", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--print-puzzles", type=int, default=3)
    parser.add_argument("--print-samples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
