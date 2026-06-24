from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConstrainedDecodeOptions:
    random_split: bool = False
    n_splits: int = 3
    split_seed: int = 0
    split_balanced: bool = False


@dataclass(frozen=True)
class ConstrainedSystem:
    check_matrix: Any
    syndrome: np.ndarray
    priors: np.ndarray
    physical_cols: int


@dataclass(frozen=True)
class _Partition:
    support: tuple[int, ...]
    blocks: tuple[tuple[int, ...], ...]


def _row_to_dense(row: Any) -> np.ndarray:
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    if sparse is not None and sparse.issparse(row):
        return np.asarray(row.todense(), dtype=np.uint8).ravel()
    return np.asarray(row, dtype=np.uint8).ravel()


def _append_row(check_matrix: Any, row: np.ndarray) -> Any:
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    if sparse is not None and sparse.issparse(check_matrix):
        row_sparse = sparse.csr_matrix(row.reshape(1, -1))
        return sparse.vstack([check_matrix, row_sparse], format="csc")
    return np.vstack([np.asarray(check_matrix), row.reshape(1, -1)])


def _rng_for_partition(
    *,
    support: np.ndarray,
    n_splits: int,
    split_seed: int,
    split_balanced: bool,
) -> np.random.Generator:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([split_seed, n_splits, int(split_balanced)], dtype=np.int64).tobytes())
    digest.update(np.asarray(support, dtype=np.int64).tobytes())
    seed = int.from_bytes(digest.digest(), byteorder="little", signed=False)
    return np.random.default_rng(seed)


def build_partition(
    obs_row: Any,
    options: ConstrainedDecodeOptions,
) -> _Partition | None:
    row = _row_to_dense(obs_row)
    support = np.flatnonzero(row).astype(np.int64)
    w = int(support.size)
    if not options.random_split or w < 2:
        return None

    m = min(max(int(options.n_splits), 2), w)
    rng = _rng_for_partition(
        support=support,
        n_splits=m,
        split_seed=int(options.split_seed),
        split_balanced=bool(options.split_balanced),
    )

    blocks: list[np.ndarray]
    if options.split_balanced:
        shuffled = support.copy()
        rng.shuffle(shuffled)
        blocks = [block for block in np.array_split(shuffled, m) if block.size]
    else:
        labels = rng.integers(0, m, size=w)
        blocks = [support[labels == label] for label in range(m)]
        blocks = [block for block in blocks if block.size]

    return _Partition(
        support=tuple(int(x) for x in support),
        blocks=tuple(tuple(int(x) for x in block) for block in blocks),
    )


def _split_check_matrix(check_matrix: Any, num_cols: int, blocks: _Partition) -> Any:
    k = len(blocks.blocks)
    try:
        from scipy import sparse
    except ImportError:
        sparse = None

    if sparse is not None and sparse.issparse(check_matrix):
        base = sparse.hstack(
            [
                check_matrix,
                sparse.csc_matrix((check_matrix.shape[0], k), dtype=np.uint8),
            ],
            format="csc",
        )
        rows = []
        for block_index, block in enumerate(blocks.blocks):
            data_cols = list(block) + [num_cols + block_index]
            data = np.ones(len(data_cols), dtype=np.uint8)
            row = sparse.csr_matrix(
                (data, ([0] * len(data_cols), data_cols)),
                shape=(1, num_cols + k),
            )
            rows.append(row)
        aux_cols = [num_cols + idx for idx in range(k)]
        rows.append(
            sparse.csr_matrix(
                (np.ones(k, dtype=np.uint8), ([0] * k, aux_cols)),
                shape=(1, num_cols + k),
            )
        )
        return sparse.vstack([base, *rows], format="csc")

    dense = np.asarray(check_matrix, dtype=np.uint8)
    base = np.pad(dense, ((0, 0), (0, k)), mode="constant")
    extra = np.zeros((k + 1, num_cols + k), dtype=np.uint8)
    for block_index, block in enumerate(blocks.blocks):
        extra[block_index, list(block)] = 1
        extra[block_index, num_cols + block_index] = 1
    extra[k, num_cols : num_cols + k] = 1
    return np.vstack([base, extra])


def build_constrained_system(
    check_matrix: Any,
    syndrome: np.ndarray,
    priors: np.ndarray,
    obs_row: Any,
    target_bit: int,
    options: ConstrainedDecodeOptions,
    partition_cache: dict[int, _Partition | None],
    observable_index: int,
) -> ConstrainedSystem:
    syndrome_arr = np.asarray(syndrome, dtype=np.uint8)
    priors_arr = np.asarray(priors, dtype=float)
    physical_cols = int(priors_arr.shape[0])

    partition = partition_cache.get(observable_index)
    if observable_index not in partition_cache:
        partition = build_partition(obs_row, options)
        partition_cache[observable_index] = partition

    if partition is None:
        row = _row_to_dense(obs_row)
        return ConstrainedSystem(
            check_matrix=_append_row(check_matrix, row),
            syndrome=np.append(syndrome_arr, int(target_bit)).astype(np.uint8),
            priors=priors_arr,
            physical_cols=physical_cols,
        )

    k = len(partition.blocks)
    return ConstrainedSystem(
        check_matrix=_split_check_matrix(check_matrix, physical_cols, partition),
        syndrome=np.concatenate(
            [
                syndrome_arr,
                np.zeros(k, dtype=np.uint8),
                np.asarray([int(target_bit)], dtype=np.uint8),
            ]
        ),
        priors=np.concatenate([priors_arr, np.full(k, 0.5, dtype=float)]),
        physical_cols=physical_cols,
    )
