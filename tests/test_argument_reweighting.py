"""Unit tests for ArgumentReweighting using a dummy DecoderAdapter."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from decoder_confidence.decoding._argument_reweighting import (
    ArgumentReweightingDecoder,
    ArgumentReweightingOptions,
    _apply_reweight,
    _clip_priors,
    _logical_from_c,
    _parse_ar_options,
)
from decoder_confidence.decoding._decoder_adapter import DecoderAdapter


# ---------------------------------------------------------------------------
# Dummy adapter
# ---------------------------------------------------------------------------


class _DummyAdapter(DecoderAdapter):
    """Adapter that returns a fixed correction vector, updated via set_priors."""

    def __init__(
        self,
        priors: np.ndarray,
        observables: np.ndarray,
        decode_sequence: list[np.ndarray],
    ) -> None:
        self._priors = _clip_priors(priors)
        self._observables = observables
        self._decode_sequence = list(decode_sequence)
        self._call_index = 0

    @property
    def priors(self) -> np.ndarray:
        return self._priors

    @property
    def check_matrix(self) -> Any:
        return np.eye(self.num_errors, dtype=int)

    @property
    def observables_matrix(self) -> Any:
        return self._observables

    @property
    def num_errors(self) -> int:
        return len(self._priors)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        idx = min(self._call_index, len(self._decode_sequence) - 1)
        result = self._decode_sequence[idx]
        self._call_index += 1
        return np.asarray(result, dtype=np.bool_)

    def set_priors(self, priors: np.ndarray) -> None:
        self._priors = _clip_priors(priors)

    def set_check_matrix(self, check_matrix: Any) -> None:
        pass

    def set_observables(self, observables_matrix: Any) -> None:
        self._observables = observables_matrix


def _make_decoder(
    decode_sequence: list[np.ndarray],
    num_errors: int = 4,
    metric: str = "ar-pec",
    test_type: str = "Ratio",
    b: float = 2.0,
    num_rounds: int = 2,
) -> ArgumentReweightingDecoder:
    priors = np.full(num_errors, 0.1)
    observables = np.eye(1, num_errors, dtype=int)
    adapter = _DummyAdapter(priors, observables, decode_sequence)
    options = ArgumentReweightingOptions(
        criterion="PEC" if "pec" in metric else "LEC",
        test_type=test_type,
        b=b,
        num_decoding_rounds=num_rounds,
    )
    return ArgumentReweightingDecoder(adapter=adapter, metric_name=metric, options=options)


# ---------------------------------------------------------------------------
# ArgumentReweightingOptions validation
# ---------------------------------------------------------------------------


def test_options_valid_ratio():
    opts = ArgumentReweightingOptions(criterion="PEC", test_type="Ratio", b=2.0, num_decoding_rounds=2)
    opts.validate()  # should not raise


def test_options_valid_gap():
    opts = ArgumentReweightingOptions(criterion="LEC", test_type="Gap", b=0.5, num_decoding_rounds=3)
    opts.validate()


def test_options_invalid_num_rounds():
    opts = ArgumentReweightingOptions(criterion="PEC", test_type="Ratio", b=2.0, num_decoding_rounds=1)
    with pytest.raises(ValueError, match="num_decoding_rounds"):
        opts.validate()


def test_options_invalid_ratio_b():
    opts = ArgumentReweightingOptions(criterion="PEC", test_type="Ratio", b=1.0, num_decoding_rounds=2)
    with pytest.raises(ValueError, match="b must be > 1"):
        opts.validate()


def test_options_invalid_gap_b():
    opts = ArgumentReweightingOptions(criterion="LEC", test_type="Gap", b=0.0, num_decoding_rounds=2)
    with pytest.raises(ValueError, match="b must be > 0"):
        opts.validate()


# ---------------------------------------------------------------------------
# _parse_ar_options
# ---------------------------------------------------------------------------


def test_parse_ar_options_pec():
    opts = _parse_ar_options("ar-pec", {"test_type": "Ratio", "b": 3.0, "num_decoding_rounds": 2})
    assert opts.criterion == "PEC"
    assert opts.test_type == "Ratio"
    assert opts.b == 3.0
    assert opts.num_decoding_rounds == 2


def test_parse_ar_options_lec_underscore():
    opts = _parse_ar_options("ar_lec", {"test_type": "Gap", "b": 1.0, "num_decoding_rounds": 3})
    assert opts.criterion == "LEC"


def test_parse_ar_options_missing_field():
    with pytest.raises(ValueError, match="test_type"):
        _parse_ar_options("ar-pec", {"b": 2.0, "num_decoding_rounds": 2})


# ---------------------------------------------------------------------------
# Reweighting functions
# ---------------------------------------------------------------------------


def test_ratio_reweight_increases_active_priors():
    priors = np.array([0.1, 0.2, 0.3, 0.4])
    opts = ArgumentReweightingOptions(criterion="PEC", test_type="Ratio", b=2.0, num_decoding_rounds=2)
    correction = np.array([1, 0, 1, 0])
    updated = _apply_reweight(priors, opts, correction)
    assert updated[0] == pytest.approx(0.1 ** 2)
    assert updated[2] == pytest.approx(0.3 ** 2)
    assert updated[1] == pytest.approx(0.2)  # unchanged
    assert updated[3] == pytest.approx(0.4)  # unchanged


def test_gap_reweight_changes_active_priors():
    priors = np.array([0.1, 0.2, 0.3, 0.4])
    opts = ArgumentReweightingOptions(criterion="PEC", test_type="Gap", b=1.0, num_decoding_rounds=2)
    correction = np.array([1, 0, 1, 0])
    updated = _apply_reweight(priors, opts, correction)
    # Inactive entries must not change
    assert updated[1] == pytest.approx(0.2)
    assert updated[3] == pytest.approx(0.4)
    # Active entries must differ from original
    assert updated[0] != pytest.approx(0.1)
    assert updated[2] != pytest.approx(0.3)


def test_ratio_reweight_empty_correction():
    priors = np.array([0.1, 0.2])
    opts = ArgumentReweightingOptions(criterion="PEC", test_type="Ratio", b=2.0, num_decoding_rounds=2)
    correction = np.array([0, 0])
    updated = _apply_reweight(priors, opts, correction)
    np.testing.assert_array_equal(updated, priors)


# ---------------------------------------------------------------------------
# _logical_from_c
# ---------------------------------------------------------------------------


def test_logical_from_c_basic():
    obs = np.array([[1, 0, 1, 0]], dtype=int)
    c = np.array([1, 0, 1, 0])
    result = _logical_from_c(obs, c)
    assert result[0] == 0  # (1+1) % 2 == 0


def test_logical_from_c_odd():
    obs = np.array([[1, 0, 0, 0]], dtype=int)
    c = np.array([1, 0, 0, 0])
    result = _logical_from_c(obs, c)
    assert result[0] == 1


def test_logical_from_c_no_observables():
    obs = np.zeros((0, 4), dtype=int)
    c = np.array([1, 0, 1, 0])
    result = _logical_from_c(obs, c)
    assert result.shape == (0,)


# ---------------------------------------------------------------------------
# Trivial syndrome shortcut
# ---------------------------------------------------------------------------


def test_trivial_syndrome_accepted():
    """c0 = all-zeros → shortcut accept without reweighting."""
    zero_c = np.zeros(4, dtype=np.bool_)
    decoder = _make_decoder([zero_c], num_errors=4)
    syndrome = np.zeros((1, 5), dtype=int)
    result = decoder.decode(syndrome)
    assert result.metrics["ar-pec"][0] == 1.0


# ---------------------------------------------------------------------------
# PEC criterion
# ---------------------------------------------------------------------------


def test_pec_accept_when_c_stable():
    """k=2 rounds: c0==c1 → accept."""
    c = np.array([1, 0, 0, 0], dtype=np.bool_)
    decoder = _make_decoder([c, c], num_errors=4, metric="ar-pec", num_rounds=2)
    syndrome = np.ones((1, 5), dtype=int)
    result = decoder.decode(syndrome)
    assert result.metrics["ar-pec"][0] == 1.0


def test_pec_reject_when_c_changes():
    """k=2 rounds: c0!=c1 → reject."""
    c0 = np.array([1, 0, 0, 0], dtype=np.bool_)
    c1 = np.array([0, 1, 0, 0], dtype=np.bool_)
    decoder = _make_decoder([c0, c1], num_errors=4, metric="ar-pec", num_rounds=2)
    syndrome = np.ones((1, 5), dtype=int)
    result = decoder.decode(syndrome)
    assert result.metrics["ar-pec"][0] == 0.0


def test_pec_three_rounds_accept():
    """k=3 rounds: c0==c1==c2 → accept."""
    c = np.array([1, 0, 0, 0], dtype=np.bool_)
    decoder = _make_decoder([c, c, c], num_errors=4, metric="ar-pec", num_rounds=3)
    syndrome = np.ones((1, 5), dtype=int)
    result = decoder.decode(syndrome)
    assert result.metrics["ar-pec"][0] == 1.0


def test_pec_three_rounds_reject_on_third():
    """k=3 rounds: c0==c1 but c2!=c0 → reject."""
    c0 = np.array([1, 0, 0, 0], dtype=np.bool_)
    c_diff = np.array([0, 0, 1, 0], dtype=np.bool_)
    decoder = _make_decoder([c0, c0, c_diff], num_errors=4, metric="ar-pec", num_rounds=3)
    syndrome = np.ones((1, 5), dtype=int)
    result = decoder.decode(syndrome)
    assert result.metrics["ar-pec"][0] == 0.0


# ---------------------------------------------------------------------------
# LEC criterion
# ---------------------------------------------------------------------------


def test_lec_accept_same_logical():
    """LEC: c0 and c1 produce same logical → accept."""
    obs = np.array([[1, 0, 0, 0]], dtype=int)
    priors = np.full(4, 0.1)
    c0 = np.array([1, 0, 0, 0], dtype=np.bool_)
    # c1 differs in error but same logical (obs@c1 = obs@c0 = 1)
    c1 = np.array([1, 1, 0, 0], dtype=np.bool_)
    adapter = _DummyAdapter(priors, obs, [c0, c1])
    options = ArgumentReweightingOptions(criterion="LEC", test_type="Ratio", b=2.0, num_decoding_rounds=2)
    decoder = ArgumentReweightingDecoder(adapter=adapter, metric_name="ar-lec", options=options)
    syndrome = np.ones((1, 4), dtype=int)
    result = decoder.decode(syndrome)
    assert result.metrics["ar-lec"][0] == 1.0


def test_lec_reject_different_logical():
    """LEC: c0 and c1 produce different logical → reject."""
    obs = np.array([[1, 0, 0, 0]], dtype=int)
    priors = np.full(4, 0.1)
    c0 = np.array([1, 0, 0, 0], dtype=np.bool_)
    c1 = np.array([0, 0, 0, 0], dtype=np.bool_)  # logical changes from 1→0
    adapter = _DummyAdapter(priors, obs, [c0, c1])
    options = ArgumentReweightingOptions(criterion="LEC", test_type="Ratio", b=2.0, num_decoding_rounds=2)
    decoder = ArgumentReweightingDecoder(adapter=adapter, metric_name="ar-lec", options=options)
    syndrome = np.ones((1, 4), dtype=int)
    result = decoder.decode(syndrome)
    assert result.metrics["ar-lec"][0] == 0.0


# ---------------------------------------------------------------------------
# Multi-shot
# ---------------------------------------------------------------------------


def test_multi_shot_mixed():
    """Two shots: first accepted (c0 stable), second rejected (c0 changes)."""
    c_stable = np.array([1, 0, 0, 0], dtype=np.bool_)
    c_diff = np.array([0, 1, 0, 0], dtype=np.bool_)
    # Sequence: shot0-round0, shot0-round1(stable), shot1-round0, shot1-round1(diff)
    priors = np.full(4, 0.1)
    obs = np.eye(1, 4, dtype=int)
    adapter = _DummyAdapter(priors, obs, [c_stable, c_stable, c_stable, c_diff])
    options = ArgumentReweightingOptions(criterion="PEC", test_type="Ratio", b=2.0, num_decoding_rounds=2)
    decoder = ArgumentReweightingDecoder(adapter=adapter, metric_name="ar-pec", options=options)
    syndromes = np.ones((2, 4), dtype=int)
    result = decoder.decode(syndromes)
    assert result.metrics["ar-pec"][0] == 1.0
    assert result.metrics["ar-pec"][1] == 0.0


# ---------------------------------------------------------------------------
# Predictions output
# ---------------------------------------------------------------------------


def test_predictions_shape():
    """Predictions must be (num_shots, num_observables)."""
    c = np.array([1, 0, 0, 0], dtype=np.bool_)
    decoder = _make_decoder([c, c], num_errors=4, num_rounds=2)
    syndromes = np.ones((3, 5), dtype=int)
    result = decoder.decode(syndromes)
    assert result.predictions.shape == (3, 1)
