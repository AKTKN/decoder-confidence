from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from decoder_confidence.decoding._relay_bp import (
    RelayBpMetricDecoder,
    _parse_relay_bp_metric_options,
)


class _FakeResult:
    def __init__(self, decoding: list[int], success: bool, iterations: int = 1) -> None:
        self.decoding = np.asarray(decoding, dtype=np.uint8)
        self.success = success
        self.iterations = iterations


class _FakeRelayBpAdapter:
    """Duck-typed stand-in for RelayBpDecoderAdapter.

    Returns queued ``_FakeResult`` objects in order, one per call to
    ``decode_detailed_single``, regardless of the syndrome/check-matrix
    passed in. This lets tests script exact stage1/stage2 outcomes.
    """

    def __init__(
        self,
        check_matrix: Any,
        observables_matrix: Any,
        priors: np.ndarray,
        responses: list[_FakeResult],
    ) -> None:
        self._check_matrix = check_matrix
        self._observables_matrix = observables_matrix
        self._priors = priors
        self._responses = list(responses)
        self.calls = 0

    @property
    def priors(self) -> np.ndarray:
        return self._priors

    @property
    def check_matrix(self) -> Any:
        return self._check_matrix

    @property
    def observables_matrix(self) -> Any:
        return self._observables_matrix

    def set_priors(self, priors: np.ndarray) -> None:
        self._priors = priors

    def set_check_matrix(self, check_matrix: Any) -> None:
        self._check_matrix = check_matrix

    def decode_detailed_single(self, syndrome: np.ndarray) -> _FakeResult:
        result = self._responses[self.calls]
        self.calls += 1
        return result


CHECK_MATRIX = np.array([[1, 0]], dtype=np.uint8)
OBSERVABLES_MATRIX = np.array([[0, 1]], dtype=np.uint8)
PRIORS = np.array([0.1, 0.1], dtype=np.float64)


def _make_decoder(metric: str, responses: list[_FakeResult], **metric_options: Any):
    adapter = _FakeRelayBpAdapter(CHECK_MATRIX, OBSERVABLES_MATRIX, PRIORS, responses)
    parsed_options = _parse_relay_bp_metric_options(metric, metric_options)
    decoder = RelayBpMetricDecoder(adapter=adapter, metric=metric, parsed_options=parsed_options)
    return decoder, adapter


@pytest.mark.parametrize(
    ("metric", "metric_options", "expected_value"),
    [
        ("linearized_logicalgap", {"forced_unconverged_confidence_value": "negative"}, -np.inf),
        ("forced_gap_ml", {"forced_unconverged_confidence_value": "negative"}, 0.0),
        (
            "reweighted_linearized_gap",
            {"b": 2.0, "forced_unconverged_confidence_value": "negative"},
            -np.inf,
        ),
    ],
)
def test_baseline_nonconvergence_forces_logical_error_and_skips_stage2(
    metric: str, metric_options: dict[str, Any], expected_value: float
) -> None:
    responses = [_FakeResult([0, 0], success=False)]
    decoder, adapter = _make_decoder(metric, responses, **metric_options)

    result = decoder.decode(np.array([[0]]))

    assert adapter.calls == 1, "stage2 must not run when baseline does not converge"
    assert result.metrics[metric][0] == expected_value
    assert result.obs_flip_idx == [[]]
    assert bool(result.metrics["__is_logical_error"][0]) is True


def test_argument_reweighting_baseline_nonconvergence_forces_logical_error() -> None:
    responses = [_FakeResult([0, 0], success=False)]
    decoder, adapter = _make_decoder(
        "argument_reweighting",
        responses,
        test_type="Ratio",
        b=2.0,
        num_decoding_rounds=2,
        criterion="LEC",
    )

    result = decoder.decode(np.array([[0]]))

    assert adapter.calls == 1
    assert bool(result.metrics["argument_reweighting"][0]) is False
    assert bool(result.metrics["__is_logical_error"][0]) is True


@pytest.mark.parametrize(
    ("forced_value", "linearized_expected", "forced_gap_ml_expected"),
    [("negative", -np.inf, 0.0), ("positive", np.inf, np.inf)],
)
def test_stage2_total_failure_uses_config_value_and_stage1_prediction(
    forced_value: str, linearized_expected: float, forced_gap_ml_expected: float
) -> None:
    # stage1 converges with correction [0, 0] -> l1 = 0; stage2 (1 forced instance) fails.
    responses = [_FakeResult([0, 0], success=True), _FakeResult([0, 0], success=False)]
    decoder, adapter = _make_decoder(
        "linearized_logicalgap",
        responses,
        forced_unconverged_confidence_value=forced_value,
    )
    result = decoder.decode(np.array([[0]]))
    assert adapter.calls == 2
    assert result.metrics["linearized_logicalgap"][0] == linearized_expected
    assert bool(result.metrics["__is_logical_error"][0]) is False
    assert result.predictions[0].tolist() == [False]

    responses = [_FakeResult([0, 0], success=True), _FakeResult([0, 0], success=False)]
    decoder, adapter = _make_decoder(
        "forced_gap_ml",
        responses,
        forced_unconverged_confidence_value=forced_value,
    )
    result = decoder.decode(np.array([[0]]))
    assert adapter.calls == 2
    assert result.metrics["forced_gap_ml"][0] == forced_gap_ml_expected
    assert bool(result.metrics["__is_logical_error"][0]) is False


def test_reweighted_linearized_gap_uses_stage2b_when_stage2a_all_fail() -> None:
    # stage1: correction [0, 0] -> l1 = 0, w1 = 0.
    # stage2a (1 constrained instance): fails to converge.
    # stage2b (reweighted unconstrained): converges with correction [0, 1] -> l_r = 1 (differs).
    weight_1 = float(np.log((1.0 - PRIORS[1]) / PRIORS[1]))
    responses = [
        _FakeResult([0, 0], success=True),  # stage1
        _FakeResult([0, 0], success=False),  # stage2a
        _FakeResult([0, 1], success=True),  # stage2b
    ]
    decoder, adapter = _make_decoder(
        "reweighted_linearized_gap",
        responses,
        b=2.0,
        forced_unconverged_confidence_value="negative",
    )

    result = decoder.decode(np.array([[0]]))

    assert adapter.calls == 3
    # gap = min(inf, w_r) - w1 = w_r - 0
    assert result.metrics["reweighted_linearized_gap"][0] == pytest.approx(weight_1)
    assert result.obs_flip_idx == [[0]]
    assert bool(result.metrics["__is_logical_error"][0]) is False


def test_reweighted_linearized_gap_total_failure_uses_config_value() -> None:
    # stage2a fails, stage2b converges but lands on the *same* logical class as baseline.
    responses = [
        _FakeResult([0, 0], success=True),  # stage1
        _FakeResult([0, 0], success=False),  # stage2a
        _FakeResult([0, 0], success=True),  # stage2b (same logical class, l_r == l1)
    ]
    decoder, adapter = _make_decoder(
        "reweighted_linearized_gap",
        responses,
        b=2.0,
        forced_unconverged_confidence_value="positive",
    )

    result = decoder.decode(np.array([[0]]))

    assert adapter.calls == 3
    assert result.metrics["reweighted_linearized_gap"][0] == np.inf
    assert bool(result.metrics["__is_logical_error"][0]) is False


@pytest.mark.parametrize(
    "metric",
    ["linearized_logicalgap", "forced_gap_ml", "reweighted_linearized_gap"],
)
def test_missing_forced_unconverged_confidence_value_raises(metric: str) -> None:
    options: dict[str, Any] = {"b": 2.0} if metric == "reweighted_linearized_gap" else {}
    with pytest.raises(ValueError, match="forced_unconverged_confidence_value"):
        _parse_relay_bp_metric_options(metric, options)


@pytest.mark.parametrize(
    "metric",
    ["linearized_logicalgap", "forced_gap_ml", "reweighted_linearized_gap"],
)
def test_invalid_forced_unconverged_confidence_value_raises(metric: str) -> None:
    options: dict[str, Any] = {"forced_unconverged_confidence_value": "sideways"}
    if metric == "reweighted_linearized_gap":
        options["b"] = 2.0
    with pytest.raises(ValueError, match="forced_unconverged_confidence_value"):
        _parse_relay_bp_metric_options(metric, options)
