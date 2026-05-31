from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.common import PuzzleDatasetMetadata
from nqweens.common import (
    EMPTY,
    PAD,
    QUEEN,
    VOCAB_SIZE,
    board_key,
    enumerate_partial_inputs,
    is_valid_solution,
    solve_nqueens,
)


@dataclass(frozen=True)
class NQueensDatasetConfig:
    output_dir: str = "nqweens/data/nqueens-8x8"
    n: int = 8
    remove_counts: Tuple[int, ...] = (5, 6, 7)
    train_fraction: float = 0.85
    seed: int = 0
    min_solutions: int = 1


def _parse_remove_counts(value: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _build_input_to_solutions(config: NQueensDatasetConfig) -> Dict[tuple[int, ...], List[tuple[int, ...]]]:
    solutions = solve_nqueens(config.n)
    input_to_solutions: Dict[tuple[int, ...], set[tuple[int, ...]]] = {}

    for solution in solutions:
        solution_key = board_key(solution)
        for partial in enumerate_partial_inputs(solution, config.n, config.remove_counts):
            input_to_solutions.setdefault(board_key(partial), set()).add(solution_key)

    filtered = {
        inp: sorted(labels)
        for inp, labels in input_to_solutions.items()
        if len(labels) >= config.min_solutions
    }
    if not filtered:
        raise ValueError("No N-Queens inputs were generated; relax min_solutions or remove_counts.")
    return filtered


def _split_inputs(
    input_to_solutions: Dict[tuple[int, ...], List[tuple[int, ...]]],
    train_fraction: float,
    seed: int,
) -> dict[str, List[tuple[int, ...]]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    keys = np.array(sorted(input_to_solutions.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    train_count = int(round(len(keys) * train_fraction))
    train_count = min(max(train_count, 1), len(keys) - 1)

    train_keys = [tuple(int(x) for x in keys[i]) for i in perm[:train_count]]
    test_keys = [tuple(int(x) for x in keys[i]) for i in perm[train_count:]]
    return {"train": sorted(train_keys), "test": sorted(test_keys)}


def _as_uint8_rows(rows: Iterable[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(list(rows), dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D token array, got shape {arr.shape}")
    return arr


def _write_split(
    output_dir: Path,
    split: str,
    input_keys: Sequence[tuple[int, ...]],
    input_to_solutions: Dict[tuple[int, ...], List[tuple[int, ...]]],
    n: int,
) -> dict:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    inputs: List[tuple[int, ...]] = []
    labels: List[tuple[int, ...]] = []
    puzzle_indices = [0]
    group_indices = [0]
    puzzle_identifiers = []
    instances = []

    for input_key in input_keys:
        solution_keys = input_to_solutions[input_key]
        for solution_key in solution_keys:
            inputs.append(input_key)
            labels.append(solution_key)
            if not is_valid_solution(solution_key, n=n, clues=input_key):
                raise AssertionError("Generated an invalid N-Queens completion")

        puzzle_identifiers.append(0)
        puzzle_indices.append(len(inputs))
        group_indices.append(len(puzzle_identifiers))
        instances.append(
            {
                "input": list(input_key),
                "solutions": [list(solution_key) for solution_key in solution_keys],
            }
        )

    input_arr = _as_uint8_rows(inputs)
    label_arr = _as_uint8_rows(labels)
    if not np.all((input_arr == EMPTY) | (input_arr == QUEEN)):
        raise AssertionError("N-Queens inputs must contain only empty and queen tokens")
    if not np.all((label_arr == EMPTY) | (label_arr == QUEEN)):
        raise AssertionError("N-Queens labels must contain only empty and queen tokens")

    results = {
        "inputs": input_arr,
        "labels": label_arr,
        "puzzle_identifiers": np.asarray(puzzle_identifiers, dtype=np.int32),
        "puzzle_indices": np.asarray(puzzle_indices, dtype=np.int32),
        "group_indices": np.asarray(group_indices, dtype=np.int32),
    }

    examples_per_puzzle = [
        int(results["puzzle_indices"][i + 1] - results["puzzle_indices"][i])
        for i in range(len(results["puzzle_indices"]) - 1)
    ]
    metadata = PuzzleDatasetMetadata(
        seq_len=n * n,
        vocab_size=VOCAB_SIZE,
        pad_id=PAD,
        ignore_label_id=PAD,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=float(np.mean(examples_per_puzzle)),
        total_puzzles=len(puzzle_identifiers),
        sets=["all"],
    )

    with (split_dir / "dataset.json").open("w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    for key, value in results.items():
        np.save(split_dir / f"all__{key}.npy", value)
    with (split_dir / "instances.json").open("w") as f:
        json.dump(instances, f)

    return {
        "split": split,
        "unique_inputs": len(input_keys),
        "examples": len(inputs),
        "min_solutions": int(np.min(examples_per_puzzle)),
        "mean_solutions": float(np.mean(examples_per_puzzle)),
        "max_solutions": int(np.max(examples_per_puzzle)),
    }


def build_dataset(config: NQueensDatasetConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_to_solutions = _build_input_to_solutions(config)
    splits = _split_inputs(input_to_solutions, config.train_fraction, config.seed)
    summaries = [
        _write_split(output_dir, split, keys, input_to_solutions, config.n)
        for split, keys in splits.items()
    ]

    all_solutions = [solution.tolist() for solution in solve_nqueens(config.n)]
    summary = {
        "config": asdict(config),
        "num_complete_solutions": len(all_solutions),
        "num_unique_inputs": len(input_to_solutions),
        "splits": summaries,
    }
    with (output_dir / "solutions.json").open("w") as f:
        json.dump(all_solutions, f)
    with (output_dir / "identifiers.json").open("w") as f:
        json.dump(["<blank>"], f)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> NQueensDatasetConfig:
    parser = argparse.ArgumentParser(description="Build the GRAM 8x8 N-Queens dataset.")
    parser.add_argument("--output-dir", default=NQueensDatasetConfig.output_dir)
    parser.add_argument("--n", type=int, default=NQueensDatasetConfig.n)
    parser.add_argument("--remove-counts", default="5,6,7")
    parser.add_argument("--train-fraction", type=float, default=NQueensDatasetConfig.train_fraction)
    parser.add_argument("--seed", type=int, default=NQueensDatasetConfig.seed)
    parser.add_argument("--min-solutions", type=int, default=NQueensDatasetConfig.min_solutions)
    args = parser.parse_args()
    return NQueensDatasetConfig(
        output_dir=args.output_dir,
        n=args.n,
        remove_counts=_parse_remove_counts(args.remove_counts),
        train_fraction=args.train_fraction,
        seed=args.seed,
        min_solutions=args.min_solutions,
    )


def main() -> None:
    summary = build_dataset(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
