from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from itertools import combinations
import json

import numpy as np


PAD = 0
EMPTY = 1
QUEEN = 2
VOCAB_SIZE = 3


@dataclass(frozen=True)
class NQueensConfig:
    n: int = 8
    remove_counts: Tuple[int, ...] = (5, 6, 7)


def solve_nqueens(n: int = 8) -> List[np.ndarray]:
    """Return all N-Queens solutions as flattened token sequences."""
    solutions: List[np.ndarray] = []
    cols = set()
    diag1 = set()
    diag2 = set()
    placement: List[int] = []

    def backtrack(row: int) -> None:
        if row == n:
            board = np.full((n, n), EMPTY, dtype=np.uint8)
            for r, c in enumerate(placement):
                board[r, c] = QUEEN
            solutions.append(board.reshape(-1))
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            placement.append(col)
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            diag2.remove(row + col)
            diag1.remove(row - col)
            cols.remove(col)
            placement.pop()

    backtrack(0)
    return solutions


def board_to_text(seq: Sequence[int], n: int = 8) -> str:
    arr = np.asarray(seq).reshape(n, n)
    rows = []
    for row in arr:
        rows.append("".join("Q" if int(x) == QUEEN else "." for x in row))
    return "\n".join(rows)


def conflicts(seq: Sequence[int], n: int = 8) -> int:
    arr = np.asarray(seq).reshape(n, n)
    positions = np.argwhere(arr == QUEEN)
    total = 0
    for i in range(len(positions)):
        r1, c1 = positions[i]
        for j in range(i + 1, len(positions)):
            r2, c2 = positions[j]
            if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                total += 1
    return total


def is_valid_solution(seq: Sequence[int], n: int = 8, clues: Sequence[int] | None = None) -> bool:
    arr = np.asarray(seq).reshape(n, n)
    if int((arr == QUEEN).sum()) != n:
        return False
    if conflicts(seq, n) != 0:
        return False
    if clues is not None:
        clue_arr = np.asarray(clues).reshape(n, n)
        if np.any((clue_arr == QUEEN) & (arr != QUEEN)):
            return False
    return True


def make_input_from_solution(solution: np.ndarray, n: int, clue_count: int, rng: np.random.Generator) -> np.ndarray:
    inp = np.full(n * n, EMPTY, dtype=np.uint8)
    if clue_count <= 0:
        return inp
    queen_indices = np.flatnonzero(solution == QUEEN)
    chosen = rng.choice(queen_indices, size=min(clue_count, queen_indices.size), replace=False)
    inp[chosen] = QUEEN
    return inp


def enumerate_partial_inputs(solution: np.ndarray, n: int, remove_counts: Sequence[int]) -> Iterable[np.ndarray]:
    """Yield partial-board inputs by removing the paper's requested queen counts."""
    queen_indices = tuple(int(x) for x in np.flatnonzero(solution == QUEEN))
    for remove_count in remove_counts:
        if remove_count < 0 or remove_count > n:
            raise ValueError(f"remove_count must be in [0, {n}], got {remove_count}")
        keep_count = n - remove_count
        for kept in combinations(queen_indices, keep_count):
            inp = np.full(n * n, EMPTY, dtype=np.uint8)
            inp[list(kept)] = QUEEN
            yield inp


def board_key(seq: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in seq)


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def unique_valid_count(seqs: Iterable[Sequence[int]], n: int = 8) -> int:
    seen = set()
    for seq in seqs:
        if is_valid_solution(seq, n):
            seen.add(tuple(int(x) for x in seq))
    return len(seen)
