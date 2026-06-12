from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._decoder_adapter import DecoderAdapter, _clip_priors
from decoder_confidence.decoding._forced_gap import (
    ForcedGapMLDecoder,
    ForcedGapMLOptions,
)
from decoder_confidence.decoding._ilp_logicalgap import _ILPLogicalGapDecoder
from decoder_confidence.decoding._linearize_logicalgap import (
    LinearizeLogicalGapDecoder,
    LinearizeLogicalGapOptions,
)


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

    assert result.metrics["stage1_weight"][0] == pytest.approx(1.0)
    assert result.metrics["stage1_obs_flip"][0].item() is True
    assert result.metrics["stage2_weight"][0] == pytest.approx(0.0)
    assert result.metrics["stage2_obs_flip"][0].item() is False
    assert result.metrics["linearize_logicalgap"][0] == pytest.approx(
        result.metrics["stage2_weight"][0] - result.metrics["stage1_weight"][0]
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

    assert result.metrics["stage1_weight"][0] == pytest.approx(0.0)
    assert result.metrics["stage1_obs_flip"][0].item() is False
    assert result.metrics["stage2_weight"][0] == pytest.approx(1.0)
    assert result.metrics["stage2_obs_flip"][0].item() is True
    assert result.metrics["stage2_2ndbest_weight"][0] == pytest.approx(2.0)
    assert result.metrics["stage2_2ndbest_obs_flip"][0].item() is True
    assert result.metrics["forced_gap_ml"][0] == pytest.approx(
        result.metrics["stage2_weight"][0] - result.metrics["stage1_weight"][0]
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


def test_ilp_logicalgap_detail_stats_use_gap_detail_metadata() -> None:
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
