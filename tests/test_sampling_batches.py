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


def test_dem_per_batch_seed_depends_on_batch_partition(tmp_path: Path) -> None:
    """Legacy behavior: changing num_batch changes the sampled population."""
    dem = _toy_circuit().detector_error_model(decompose_errors=False)

    single = tmp_path / "single"
    split = tmp_path / "split"

    sample_batches_from_dem(
        dem, single, [9], det_sample_seed=123, sampling_method="per_batch_seed"
    )
    sample_batches_from_dem(
        dem, split, [2, 3, 4], det_sample_seed=123, sampling_method="per_batch_seed"
    )

    assert _concat_batches(single) != _concat_batches(split)


def test_dem_per_batch_seed_matches_manual_per_batch_sampler(tmp_path: Path) -> None:
    """Reproduces the exact pre-2026-06-25 seed=det_sample_seed+(batch_index-1) scheme."""
    dem = _toy_circuit().detector_error_model(decompose_errors=False)

    out_dir = tmp_path / "legacy"
    sample_batches_from_dem(
        dem, out_dir, [2, 3], det_sample_seed=10, sampling_method="per_batch_seed"
    )

    expected = b""
    for batch_index, shots in enumerate([2, 3], start=1):
        sampler = dem.compile_sampler(seed=10 + (batch_index - 1))
        dets, obs, _ = sampler.sample(shots)
        import numpy as np

        data = np.concatenate([dets, obs], axis=1)
        num_bits = data.shape[1]
        bytes_per_shot = (num_bits + 7) // 8
        pad = bytes_per_shot * 8 - num_bits
        if pad:
            data = np.concatenate(
                [data, np.zeros((data.shape[0], pad), dtype=data.dtype)], axis=1
            )
        expected += np.packbits(data.astype(np.uint8), axis=1, bitorder="little").tobytes()

    assert _concat_batches(out_dir) == expected


def test_invalid_sampling_method_raises(tmp_path: Path) -> None:
    dem = _toy_circuit().detector_error_model(decompose_errors=False)
    try:
        sample_batches_from_dem(
            dem, tmp_path / "bad", [9], det_sample_seed=1, sampling_method="bogus"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid sampling_method")
