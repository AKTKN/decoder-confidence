from __future__ import annotations

import sys
from pathlib import Path

import stim

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoder_confidence.sampling.sampler import sample_batches, sample_batches_from_dem


def _toy_circuit() -> stim.Circuit:
    return stim.Circuit(
        """
        X_ERROR(0.2) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )


def _concat_batches(path: Path) -> bytes:
    return b"".join(
        batch.read_bytes()
        for batch in sorted(path.glob("det_batch=*.b8"))
    )


def test_dem_sampling_is_independent_of_batch_partition(tmp_path: Path) -> None:
    dem = _toy_circuit().detector_error_model(decompose_errors=False)

    single = tmp_path / "single"
    split = tmp_path / "split"

    sample_batches_from_dem(dem, single, [9], det_sample_seed=123)
    sample_batches_from_dem(dem, split, [2, 3, 4], det_sample_seed=123)

    assert _concat_batches(single) == _concat_batches(split)


def test_circuit_sampling_is_independent_of_batch_partition(tmp_path: Path) -> None:
    circuit = _toy_circuit()

    single = tmp_path / "single"
    split = tmp_path / "split"

    sample_batches(circuit, single, [9], det_sample_seed=123)
    sample_batches(circuit, split, [2, 3, 4], det_sample_seed=123)

    assert _concat_batches(single) == _concat_batches(split)
