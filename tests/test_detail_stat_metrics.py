from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._constraints import (
    ConstrainedDecodeOptions,
    build_constrained_system,
)
from decoder_confidence.decoding._decoder_adapter import DecoderAdapter, _clip_priors
from decoder_confidence.decoding._forced_gap import (
    ForcedGapMLDecoder,
    ForcedGapMLOptions,
)
from decoder_confidence.decoding._linearize_logicalgap import (
    LinearizeLogicalGapDecoder,
    LinearizeLogicalGapOptions,
)
from decoder_confidence.decoding.result_collection import collect_results
from decoder_confidence.execution.worker import DECODER_STAT_PREFIX, DETAIL_STAT_PREFIX


def _priors_for_weights(weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return 1.0 / (1.0 + np.exp(w))


class _SequenceAdapter(DecoderAdapter):
    def __init__(
        self,
        priors: np.ndarray,
        observables: np.ndarray,
        decode_sequence: list[np.ndarray],
    ) -> None:
        self._priors = _clip_priors(priors)
        self._observables = observables
        self._check_matrix = np.eye(len(priors), dtype=int)
        self._decode_sequence = list(decode_sequence)
        self._call_index = 0

    @property
    def priors(self) -> np.ndarray:
        return self._priors

    @property
    def check_matrix(self) -> Any:
        return self._check_matrix

    @property
    def observables_matrix(self) -> Any:
        return self._observables

    @property
    def num_errors(self) -> int:
        return len(self._priors)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        idx = min(self._call_index, len(self._decode_sequence) - 1)
        self._call_index += 1
        return np.asarray(self._decode_sequence[idx], dtype=np.bool_)

    def set_priors(self, priors: np.ndarray) -> None:
        self._priors = _clip_priors(priors)

    def set_check_matrix(self, check_matrix: Any) -> None:
        self._check_matrix = check_matrix

    def set_observables(self, observables_matrix: Any) -> None:
        self._observables = observables_matrix


class _BruteForceAdapter(DecoderAdapter):
    def __init__(self, priors: np.ndarray, check_matrix: np.ndarray, observables: np.ndarray) -> None:
        self._priors = _clip_priors(priors)
        self._check_matrix = np.asarray(check_matrix, dtype=np.uint8)
        self._observables = np.asarray(observables, dtype=np.uint8)

    @property
    def priors(self) -> np.ndarray:
        return self._priors

    @property
    def check_matrix(self) -> Any:
        return self._check_matrix

    @property
    def observables_matrix(self) -> Any:
        return self._observables

    @property
    def num_errors(self) -> int:
        return int(self._check_matrix.shape[1])

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        n = int(self._check_matrix.shape[1])
        for mask in range(1 << n):
            candidate = np.asarray([(mask >> bit) & 1 for bit in range(n)], dtype=np.uint8)
            if np.array_equal((self._check_matrix @ candidate) % 2, syndrome):
                return candidate.astype(np.bool_)
        raise AssertionError("no feasible correction")

    def set_priors(self, priors: np.ndarray) -> None:
        self._priors = _clip_priors(priors)

    def set_check_matrix(self, check_matrix: Any) -> None:
        self._check_matrix = np.asarray(check_matrix.todense() if hasattr(check_matrix, "todense") else check_matrix, dtype=np.uint8)

    def set_observables(self, observables_matrix: Any) -> None:
        self._observables = np.asarray(observables_matrix, dtype=np.uint8)


def test_linearize_logicalgap_detail_stats_reconstruct_metric() -> None:
    adapter = _SequenceAdapter(
        priors=_priors_for_weights([1.0, 2.0, 4.0]),
        observables=np.array([[1, 0, 0], [0, 1, 0]], dtype=int),
        decode_sequence=[
            np.array([1, 0, 0], dtype=int),
            np.array([0, 0, 0], dtype=int),
            np.array([1, 1, 0], dtype=int),
        ],
    )
    decoder = LinearizeLogicalGapDecoder(
        adapter=adapter,
        options=LinearizeLogicalGapOptions(get_detail_stat=True),
    )

    result = decoder.decode(
        np.array([[0, 0, 0]], dtype=int),
        true_obs=np.array([[0, 0]], dtype=bool),
    )

    assert result.detail_stats["stage1_weight"][0] == pytest.approx(1.0)
    assert result.detail_stats["stage1_obs_flip"][0].item() is True
    assert result.detail_stats["stage2_weight"][0] == pytest.approx(0.0)
    assert result.detail_stats["stage2_obs_flip"][0].item() is False
    assert result.metrics["linearize_logicalgap"][0] == pytest.approx(
        result.detail_stats["stage2_weight"][0] - result.detail_stats["stage1_weight"][0]
    )


def test_forced_gap_ml_detail_stats_reconstruct_metric_and_second_stage2() -> None:
    adapter = _SequenceAdapter(
        priors=_priors_for_weights([1.0, 2.0, 4.0]),
        observables=np.array([[1, 0, 0], [0, 1, 0]], dtype=int),
        decode_sequence=[
            np.array([0, 0, 0], dtype=int),
            np.array([1, 0, 0], dtype=int),
            np.array([0, 1, 0], dtype=int),
        ],
    )
    decoder = ForcedGapMLDecoder(
        adapter=adapter,
        options=ForcedGapMLOptions(get_detail_stat=True),
    )

    result = decoder.decode(
        np.array([[0, 0, 0]], dtype=int),
        true_obs=np.array([[0, 0]], dtype=bool),
    )

    assert result.detail_stats["stage1_weight"][0] == pytest.approx(0.0)
    assert result.detail_stats["stage1_obs_flip"][0].item() is False
    assert result.detail_stats["stage2_weight"][0] == pytest.approx(1.0)
    assert result.detail_stats["stage2_obs_flip"][0].item() is True
    assert result.detail_stats["stage2_2ndbest_weight"][0] == pytest.approx(2.0)
    assert result.detail_stats["stage2_2ndbest_obs_flip"][0].item() is True
    assert result.metrics["forced_gap_ml"][0] == pytest.approx(
        result.detail_stats["stage2_weight"][0] - result.detail_stats["stage1_weight"][0]
    )


class _FakeILPShot:
    def __init__(self) -> None:
        self.predicted_observables = np.array([1, 0], dtype=int)
        self.metadata = {
            "logical_gap": 2.0,
            "obs_flip_idx": [0],
            "gap_detail": {
                "stage1_weight": 3.0,
                "stage2_weight": 5.0,
                "stage2_error_vector": np.array([0, 1, 0], dtype=int),
            },
        }


class _FakeILPDecoder:
    _observables = np.array([[1, 0, 0], [0, 1, 0]], dtype=int)

    def decode_batch_result(self, syndromes: np.ndarray, **kwargs: Any) -> list[_FakeILPShot]:
        assert kwargs["get_logicalgap"] is True
        assert kwargs["get_gap_detail"] is True
        return [_FakeILPShot() for _ in range(syndromes.shape[0])]


@pytest.mark.skip(reason="ILP validation is excluded from this non-ILP test suite")
def test_ilp_logicalgap_detail_stats_use_gap_detail_metadata() -> None:
    from decoder_confidence.decoding._ilp_logicalgap import _ILPLogicalGapDecoder

    decoder = _ILPLogicalGapDecoder(
        decoder=_FakeILPDecoder(),  # type: ignore[arg-type]
        get_detail_stat=True,
    )

    result = decoder.decode(
        np.array([[0, 0]], dtype=int),
        true_obs=np.array([[0, 0]], dtype=bool),
    )

    assert isinstance(result, DecodingResult)
    assert result.metrics["stage1_weight"][0] == pytest.approx(3.0)
    assert result.metrics["stage1_obs_flip"][0].item() is True
    assert result.metrics["stage2_weight"][0] == pytest.approx(5.0)
    assert result.metrics["stage2_obs_flip"][0].item() is True
    assert result.metrics["logical_gap"][0] == pytest.approx(
        result.metrics["stage2_weight"][0] - result.metrics["stage1_weight"][0]
    )


def test_collect_results_writes_batch_detail_decoder_stats_and_converts_legacy(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    pl.DataFrame({"shot_id": [20], "stage1_weight": [7.0]}).write_parquet(
        output_dir / "metric=stage1_weight_batch=3.parquet"
    )
    pl.DataFrame({"shot_id": [20], "stage1_obs_flip": [True]}).write_parquet(
        output_dir / "metric=stage1_obs_flip_batch=3.parquet"
    )
    pl.DataFrame({"shot_id": [20], "stage2_weight": [8.0]}).write_parquet(
        output_dir / "metric=stage2_weight_batch=3.parquet"
    )
    chunk_dir = output_dir / "chunks" / "batch=1"
    chunk_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "shot_id": [0, 1],
            "is_logical_error": [False, True],
            "metric_forced_gap_ml": [0.2, 0.8],
            f"{DETAIL_STAT_PREFIX}stage1_obs_flip": [False, True],
            f"{DETAIL_STAT_PREFIX}stage1_weight": [1.0, 2.0],
            f"{DETAIL_STAT_PREFIX}stage2_obs_flip": [True, False],
            f"{DETAIL_STAT_PREFIX}stage2_weight": [3.0, 4.0],
            f"{DETAIL_STAT_PREFIX}stage2_2ndbest_obs_flip": [False, False],
            f"{DETAIL_STAT_PREFIX}stage2_2ndbest_weight": [5.0, 6.0],
            f"{DECODER_STAT_PREFIX}baseline_iteration": [2.0, 3.0],
            f"{DECODER_STAT_PREFIX}forced_iteration": [4.0, 5.0],
            f"{DECODER_STAT_PREFIX}baseline_cluster_llr": [0.1, 0.2],
            f"{DECODER_STAT_PREFIX}forced_cluster_llr": [0.3, 0.4],
        }
    ).write_parquet(chunk_dir / "chunk_000.parquet")

    metric_names = collect_results(chunk_dir, output_dir, 1, "forced_gap_ml")

    assert metric_names == ["forced_gap_ml"]
    assert not (output_dir / "metric=stage1_weight_batch=1.parquet").exists()
    detail = pl.read_parquet(output_dir / "detailed_stats_batch=1.parquet")
    assert detail.columns == [
        "shot_id",
        "baseline_logical_error",
        "baseline_correction_weight",
        "forced_2nd_best_logical_error",
        "forced_2nd_best_correction_weight",
        "forced_logical_error",
        "forced_correction_weight",
    ]
    assert detail["baseline_correction_weight"].to_list() == [1.0, 2.0]
    decoder_stat = pl.read_parquet(output_dir / "decoder_stat_batch=1.parquet")
    assert not (output_dir / "decoder_stat.parquet").exists()
    assert decoder_stat.select(
        ["shot_id", "baseline_iteration", "baseline_cluster_llr"]
    ).to_dict(as_series=False) == {
        "shot_id": [0, 1],
        "baseline_iteration": [2.0, 3.0],
        "baseline_cluster_llr": [0.1, 0.2],
    }
    converted = pl.read_parquet(output_dir / "detailed_stats_batch=3.parquet")
    assert converted.to_dict(as_series=False) == {
        "shot_id": [20],
        "baseline_correction_weight": [7.0],
        "baseline_logical_error": [True],
        "forced_correction_weight": [8.0],
    }
    assert (output_dir / "metric=stage1_weight_batch=3.parquet").exists()

    chunk_dir2 = output_dir / "chunks" / "batch=2"
    chunk_dir2.mkdir(parents=True)
    pl.DataFrame(
        {
            "shot_id": [10],
            "is_logical_error": [False],
            "metric_forced_gap_ml": [0.1],
            f"{DECODER_STAT_PREFIX}baseline_iteration": [9.0],
        }
    ).write_parquet(chunk_dir2 / "chunk_000.parquet")
    collect_results(chunk_dir2, output_dir, 2, "forced_gap_ml")
    batch2 = pl.read_parquet(output_dir / "decoder_stat_batch=2.parquet")
    assert batch2.to_dict(as_series=False) == {
        "shot_id": [10],
        "baseline_iteration": [9.0],
    }


def test_random_split_constrained_system_preserves_physical_parity() -> None:
    check = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    priors = np.array([0.1, 0.2, 0.3, 0.4])
    obs_row = np.array([1, 1, 1, 1], dtype=np.uint8)
    syndrome = np.array([1, 0], dtype=np.uint8)

    for n_splits in [2, 3, 10]:
        constrained = build_constrained_system(
            check,
            syndrome,
            priors,
            obs_row,
            1,
            ConstrainedDecodeOptions(random_split=True, n_splits=n_splits, split_seed=17),
            {},
            0,
        )
        adapter = _BruteForceAdapter(constrained.priors, constrained.check_matrix, obs_row.reshape(1, -1))
        correction = adapter.decode(constrained.syndrome)
        physical = correction[: constrained.physical_cols].astype(np.uint8)
        assert int(obs_row @ physical % 2) == 1


def test_random_split_and_detail_stats_execute_together() -> None:
    adapter = _BruteForceAdapter(
        priors=_priors_for_weights([1.0, 2.0, 3.0, 4.0]),
        check_matrix=np.array([[1, 0, 1, 0]], dtype=np.uint8),
        observables=np.array([[1, 1, 1, 1]], dtype=np.uint8),
    )
    decoder = LinearizeLogicalGapDecoder(
        adapter=adapter,
        options=LinearizeLogicalGapOptions(
            get_detail_stat=True,
            random_split=True,
            n_splits=3,
            split_seed=5,
        ),
    )

    result = decoder.decode(
        np.array([[0]], dtype=np.uint8),
        true_obs=np.array([[0]], dtype=bool),
    )

    assert "linearize_logicalgap" in result.metrics
    assert result.detail_stats["stage1_weight"].shape == (1,)
    assert result.detail_stats["stage2_weight"].shape == (1,)
    assert result.decoder_stats == {}
