from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SRC_ROOT = REPO_ROOT / "analysis" / "src"
if str(ANALYSIS_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC_ROOT))

from utils.length4_cycle import Length4CycleAnalyzer, QuantumCircuitParameters


class _SyntheticAnalyzer(Length4CycleAnalyzer):
    def __init__(self, check_matrix: np.ndarray, observable_matrix: np.ndarray) -> None:
        super().__init__(
            QuantumCircuitParameters(
                code="synthetic_Z",
                d=1,
                rounds=1,
                noise_model="synthetic",
                p=0.0,
            )
        )
        self._synthetic_check = np.asarray(check_matrix, dtype=np.uint8)
        self._synthetic_observable = np.asarray(observable_matrix, dtype=np.uint8)

    @property
    def check_matrix(self):
        return self._synthetic_check

    @property
    def observable_matrix(self):
        return self._synthetic_observable

    def load_matrices(self):
        self._check_matrix_csr = None
        from scipy import sparse

        self._check_matrix_csr = sparse.csr_matrix(self._synthetic_check)
        return None


def test_unsplit_cycle_count_sums_pair_overlaps_per_original_check() -> None:
    analyzer = _SyntheticAnalyzer(
        check_matrix=np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.uint8,
        ),
        observable_matrix=np.array([[1, 1, 1, 0]], dtype=np.uint8),
    )

    result = analyzer.analyze_observable(0)

    assert result.support_size == 3
    assert result.cycle_count == 2


def test_random_split_cycle_count_uses_each_constraint_block() -> None:
    analyzer = _SyntheticAnalyzer(
        check_matrix=np.array([[1, 1, 1, 1]], dtype=np.uint8),
        observable_matrix=np.array([[1, 1, 1, 1]], dtype=np.uint8),
    )

    unsplit = analyzer.analyze_observable(0)
    split = analyzer.analyze_random_split(
        0,
        n_splits=2,
        split_seed=7,
        split_balanced=True,
    )

    assert unsplit.cycle_count == 6
    assert split.block_sizes == (2, 2)
    assert split.cycle_count == 2


def test_monte_carlo_rejects_empty_sample_count() -> None:
    analyzer = _SyntheticAnalyzer(
        check_matrix=np.eye(3, dtype=np.uint8),
        observable_matrix=np.array([[1, 1, 0]], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="sample_count"):
        analyzer.monte_carlo_random_split(0, sample_count=0)
