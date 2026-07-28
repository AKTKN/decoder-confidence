"""Non-convergence handling for relay-bp, exercised against the real decoder classes.

These scenarios used to be tested against a separate, dead-code implementation in the
now-deleted ``_relay_bp.py`` (see ``documents/relay_bp_nonconvergence_behavior.md``).
That module was never actually reachable from ``decoder_factory.py`` for
``linearize_logicalgap``/``forced_gap_ml``/``reweighted_linearized_gap`` because those
metric names are intercepted earlier in the dispatch chain regardless of the configured
decoder. The non-convergence handling now lives directly in the real, decoder-agnostic
implementations, so these tests exercise those instead.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from decoder_confidence.decoding._argument_reweighting import (
    ArgumentReweightingDecoder,
    ArgumentReweightingOptions,
)
from decoder_confidence.decoding._decoder_adapter import RelayBpDecoderAdapter
from decoder_confidence.decoding._forced_gap import (
    ForcedGapMLDecoder,
    ForcedGapMLOptions,
    _parse_forced_gap_ml_options,
)
from decoder_confidence.decoding._linearize_logicalgap import (
    LinearizeLogicalGapDecoder,
    LinearizeLogicalGapOptions,
    _parse_linearize_options,
)
from decoder_confidence.decoding._reweighted_linearized_gap import (
    ReweightedLinearizedGapDecoder,
    ReweightedLinearizedGapOptions,
    _parse_rlg_options,
)


class _FakeResult:
    def __init__(self, decoding: list[int], success: bool, iterations: int = 1) -> None:
        self.decoding = np.asarray(decoding, dtype=np.uint8)
        self.success = success
        self.iterations = iterations


class _FakeRelayDecoder:
    """Returns queued ``_FakeResult`` objects in order, one per ``decode_detailed`` call,
    regardless of the syndrome passed in. Lets tests script exact stage outcomes."""

    def __init__(self, responses: list[_FakeResult]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def decode_detailed(self, syndrome_u8: np.ndarray) -> _FakeResult:
        result = self._responses[self.calls]
        self.calls += 1
        return result


CHECK_MATRIX = np.array([[1, 0]], dtype=np.uint8)
OBSERVABLES_MATRIX = np.array([[0, 1]], dtype=np.uint8)
PRIORS = np.array([0.1, 0.1], dtype=np.float64)


def _make_relay_bp_adapter(
    responses: list[_FakeResult],
) -> tuple[RelayBpDecoderAdapter, _FakeRelayDecoder]:
    fake_decoder = _FakeRelayDecoder(responses)
    adapter = RelayBpDecoderAdapter(
        _check_matrix=CHECK_MATRIX,
        _observables_matrix=OBSERVABLES_MATRIX,
        _priors=PRIORS,
        _decoder_options={},
        _decoder=fake_decoder,
    )
    # set_priors/set_check_matrix(_and_priors) call self._rebuild(), which would
    # otherwise try to construct a real relay_bp Rust decoder.
    adapter._rebuild = lambda: None  # type: ignore[method-assign]
    return adapter, fake_decoder


# ---------------------------------------------------------------------------
# linearize_logicalgap
# ---------------------------------------------------------------------------


def test_linearize_logicalgap_baseline_nonconvergence_forces_logical_error() -> None:
    adapter, fake = _make_relay_bp_adapter([_FakeResult([0, 0], success=False)])
    decoder = LinearizeLogicalGapDecoder(
        adapter=adapter,
        options=LinearizeLogicalGapOptions(forced_unconverged_confidence_value="negative"),
    )

    result = decoder.decode(np.array([[0]]))

    assert fake.calls == 1, "stage2 must not run when baseline does not converge"
    assert result.metrics["linearize_logicalgap"][0] == -np.inf
    assert result.obs_flip_idx == [[]]
    assert bool(result.metrics["__is_logical_error"][0]) is True


@pytest.mark.parametrize(
    ("forced_value", "expected"), [("negative", -np.inf), ("positive", np.inf)]
)
def test_linearize_logicalgap_stage2_total_failure_uses_config_value(
    forced_value: str, expected: float
) -> None:
    adapter, fake = _make_relay_bp_adapter(
        [_FakeResult([0, 0], success=True), _FakeResult([0, 0], success=False)]
    )
    decoder = LinearizeLogicalGapDecoder(
        adapter=adapter,
        options=LinearizeLogicalGapOptions(forced_unconverged_confidence_value=forced_value),
    )

    result = decoder.decode(np.array([[0]]))

    assert fake.calls == 2
    assert result.metrics["linearize_logicalgap"][0] == expected
    assert bool(result.metrics["__is_logical_error"][0]) is False
    assert result.predictions[0].tolist() == [False]


# ---------------------------------------------------------------------------
# forced_gap_ml
# ---------------------------------------------------------------------------


def test_forced_gap_ml_baseline_nonconvergence_forces_logical_error() -> None:
    adapter, fake = _make_relay_bp_adapter([_FakeResult([0, 0], success=False)])
    decoder = ForcedGapMLDecoder(
        adapter=adapter,
        options=ForcedGapMLOptions(forced_unconverged_confidence_value="negative"),
    )

    result = decoder.decode(np.array([[0]]))

    assert fake.calls == 1
    assert result.metrics["forced_gap_ml"][0] == 0.0
    assert bool(result.metrics["__is_logical_error"][0]) is True


@pytest.mark.parametrize(("forced_value", "expected"), [("negative", 0.0), ("positive", np.inf)])
def test_forced_gap_ml_stage2_total_failure_uses_config_value(
    forced_value: str, expected: float
) -> None:
    adapter, fake = _make_relay_bp_adapter(
        [_FakeResult([0, 0], success=True), _FakeResult([0, 0], success=False)]
    )
    decoder = ForcedGapMLDecoder(
        adapter=adapter,
        options=ForcedGapMLOptions(forced_unconverged_confidence_value=forced_value),
    )

    result = decoder.decode(np.array([[0]]))

    assert fake.calls == 2
    assert result.metrics["forced_gap_ml"][0] == expected
    assert bool(result.metrics["__is_logical_error"][0]) is False


# ---------------------------------------------------------------------------
# reweighted_linearized_gap
# ---------------------------------------------------------------------------


def test_reweighted_linearized_gap_baseline_nonconvergence_forces_logical_error() -> None:
    adapter, fake = _make_relay_bp_adapter([_FakeResult([0, 0], success=False)])
    decoder = ReweightedLinearizedGapDecoder(
        adapter=adapter,
        options=ReweightedLinearizedGapOptions(
            b=2.0, forced_unconverged_confidence_value="negative"
        ),
    )

    result = decoder.decode(np.array([[0]]))

    assert fake.calls == 1, "stage2a/2b must not run when baseline does not converge"
    assert result.metrics["reweighted_linearized_gap"][0] == -np.inf
    assert result.obs_flip_idx == [[]]
    assert bool(result.metrics["__is_logical_error"][0]) is True


def test_reweighted_linearized_gap_stage2a_empty_uses_stage2b_diff_class() -> None:
    # stage1 converges to l1=0; stage2a (1 forced instance) fails; stage2b converges
    # to a different logical class (l_r=1). This is a regression test: the fix must
    # not fall back to the config sentinel just because stage2a produced nothing.
    adapter, fake = _make_relay_bp_adapter(
        [
            _FakeResult([0, 0], success=True),
            _FakeResult([0, 0], success=False),
            _FakeResult([1, 1], success=True),
        ]
    )
    decoder = ReweightedLinearizedGapDecoder(
        adapter=adapter,
        options=ReweightedLinearizedGapOptions(
            b=2.0, forced_unconverged_confidence_value="negative"
        ),
    )

    result = decoder.decode(np.array([[0]]))

    assert fake.calls == 3
    expected_gap = 2 * np.log(9)  # w_r - w1, weights derived from priors=[0.1, 0.1]
    assert result.metrics["reweighted_linearized_gap"][0] == pytest.approx(expected_gap)
    assert result.obs_flip_idx == [[0]]
    assert bool(result.metrics["__is_logical_error"][0]) is False


def test_reweighted_linearized_gap_stage2a_empty_and_stage2b_same_class_uses_config_value() -> (
    None
):
    # stage2a fails and stage2b converges back to the *same* logical class as stage1:
    # no usable differing-class candidate exists at all -> config sentinel applies.
    adapter, fake = _make_relay_bp_adapter(
        [
            _FakeResult([0, 0], success=True),
            _FakeResult([0, 0], success=False),
            _FakeResult([0, 0], success=True),
        ]
    )
    decoder = ReweightedLinearizedGapDecoder(
        adapter=adapter,
        options=ReweightedLinearizedGapOptions(
            b=2.0, forced_unconverged_confidence_value="negative"
        ),
    )

    result = decoder.decode(np.array([[0]]))

    assert fake.calls == 3
    assert result.metrics["reweighted_linearized_gap"][0] == -np.inf
    assert result.obs_flip_idx == [[]]
    assert bool(result.metrics["__is_logical_error"][0]) is False


# ---------------------------------------------------------------------------
# argument_reweighting (ArgumentReweightingDecoder)
# ---------------------------------------------------------------------------


def test_argument_reweighting_round0_nonconvergence_forces_logical_error() -> None:
    adapter, fake = _make_relay_bp_adapter([_FakeResult([0, 0], success=False)])
    decoder = ArgumentReweightingDecoder(
        adapter=adapter,
        metric_name="argument_reweighting",
        options=ArgumentReweightingOptions(
            criterion="LEC", test_type="Ratio", b=2.0, num_decoding_rounds=2
        ),
    )

    result = decoder.decode(np.array([[0]]))

    assert fake.calls == 1
    assert bool(result.metrics["argument_reweighting"][0]) is False
    assert bool(result.metrics["__is_logical_error"][0]) is True


# ---------------------------------------------------------------------------
# forced_unconverged_confidence_value requiredness / validation
# ---------------------------------------------------------------------------


def test_missing_forced_unconverged_confidence_value_raises_for_relay_bp() -> None:
    adapter, _ = _make_relay_bp_adapter([])
    with pytest.raises(ValueError, match="forced_unconverged_confidence_value"):
        LinearizeLogicalGapDecoder(adapter=adapter, options=LinearizeLogicalGapOptions())

    adapter, _ = _make_relay_bp_adapter([])
    with pytest.raises(ValueError, match="forced_unconverged_confidence_value"):
        ForcedGapMLDecoder(adapter=adapter, options=ForcedGapMLOptions())

    adapter, _ = _make_relay_bp_adapter([])
    with pytest.raises(ValueError, match="forced_unconverged_confidence_value"):
        ReweightedLinearizedGapDecoder(
            adapter=adapter, options=ReweightedLinearizedGapOptions(b=2.0)
        )


def test_invalid_forced_unconverged_confidence_value_raises() -> None:
    with pytest.raises(ValueError, match="forced_unconverged_confidence_value"):
        _parse_linearize_options({"forced_unconverged_confidence_value": "bogus"})

    with pytest.raises(ValueError, match="forced_unconverged_confidence_value"):
        _parse_forced_gap_ml_options({"forced_unconverged_confidence_value": "bogus"})

    with pytest.raises(ValueError, match="forced_unconverged_confidence_value"):
        _parse_rlg_options({"b": 2.0, "forced_unconverged_confidence_value": "bogus"})


def test_forced_unconverged_confidence_value_accepted_as_metric_option() -> None:
    opts = _parse_linearize_options({"forced_unconverged_confidence_value": "Positive"})
    assert opts.forced_unconverged_confidence_value == "positive"

    opts2 = _parse_forced_gap_ml_options({"forced_unconverged_confidence_value": "Negative"})
    assert opts2.forced_unconverged_confidence_value == "negative"

    opts3 = _parse_rlg_options({"b": 2.0, "forced_unconverged_confidence_value": "positive"})
    assert opts3.forced_unconverged_confidence_value == "positive"
