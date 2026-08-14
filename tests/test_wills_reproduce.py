"""Tests for the ``wills_reproduce`` metric (``_wills_reproduce.py``).

Reproduces the "forced gap" post-selection strategy of Wills, Yoder &
Chuang, "Forced Gap Post-Selection for Quantum LDPC Codes and their
Operations" (arXiv:2605.20346). Uses the same fake-decoder-stubbing
approach as ``test_relay_bp_nonconvergence.py``, but ``wills_reproduce``
always needs two adapters (baseline vs. forced runs, since the paper
differentiates their ``num_sets``/R -- see Appendix A), unlike
``forced_gap_ml`` which reuses a single adapter for both stages.
"""
from __future__ import annotations

import numpy as np
import pytest

from decoder_confidence.decoding._decoder_adapter import RelayBpDecoderAdapter
from decoder_confidence.decoding._wills_reproduce import (
    WillsReproduceDecoder,
    WillsReproduceOptions,
    _parse_wills_reproduce_options,
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


def test_baseline_nonconvergence_is_erasure() -> None:
    baseline, baseline_fake = _make_relay_bp_adapter([_FakeResult([0, 0], success=False)])
    forced, forced_fake = _make_relay_bp_adapter([])
    decoder = WillsReproduceDecoder(
        baseline_adapter=baseline,
        forced_adapter=forced,
        options=WillsReproduceOptions(forced_num_sets=25),
    )

    result = decoder.decode(np.array([[0]]))

    assert baseline_fake.calls == 1
    assert forced_fake.calls == 0, "forced runs must not run when baseline does not converge"
    assert result.metrics["wills_reproduce"][0] == 0.0
    assert bool(result.metrics["__is_logical_error"][0]) is True
    assert result.obs_flip_idx == [[]]


def test_all_forced_runs_unconverged_gives_infinite_gap() -> None:
    baseline, baseline_fake = _make_relay_bp_adapter([_FakeResult([0, 0], success=True)])
    forced, forced_fake = _make_relay_bp_adapter([_FakeResult([0, 0], success=False)])
    decoder = WillsReproduceDecoder(
        baseline_adapter=baseline,
        forced_adapter=forced,
        options=WillsReproduceOptions(forced_num_sets=25),
    )

    result = decoder.decode(np.array([[0]]))

    assert baseline_fake.calls == 1
    assert forced_fake.calls == 1  # K=1 observable -> 1 forced run
    assert result.metrics["wills_reproduce"][0] == np.inf
    assert bool(result.metrics["__is_logical_error"][0]) is False
    assert result.predictions[0].tolist() == [False]


def test_forced_run_convergence_gives_finite_gap() -> None:
    baseline, baseline_fake = _make_relay_bp_adapter([_FakeResult([0, 0], success=True)])
    forced, forced_fake = _make_relay_bp_adapter([_FakeResult([1, 1], success=True)])
    decoder = WillsReproduceDecoder(
        baseline_adapter=baseline,
        forced_adapter=forced,
        options=WillsReproduceOptions(forced_num_sets=25),
    )

    result = decoder.decode(np.array([[0]]))

    assert baseline_fake.calls == 1
    assert forced_fake.calls == 1
    # weights derived from priors=[0.1, 0.1]: w = sum(c_i * log((1-p_i)/p_i)).
    # baseline correction [0, 0] -> w0 = 0; forced correction [1, 1] -> w_i = 2*log(9).
    expected_gap = 2 * np.log(9)
    assert result.metrics["wills_reproduce"][0] == pytest.approx(expected_gap)
    assert bool(result.metrics["__is_logical_error"][0]) is False
    # baseline has the lower weight, so it stays the ML pick (predicted class [0]).
    assert result.predictions[0].tolist() == [False]
    assert result.obs_flip_idx == [[]]


def test_missing_forced_num_sets_raises() -> None:
    with pytest.raises(ValueError, match="forced_num_sets"):
        _parse_wills_reproduce_options({})


def test_forced_num_sets_parsed() -> None:
    opts = _parse_wills_reproduce_options({"forced_num_sets": 25, "get_detail_stat": True})
    assert opts.forced_num_sets == 25
    assert opts.get_detail_stat is True


def test_unknown_metric_option_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported wills_reproduce metric option"):
        _parse_wills_reproduce_options({"forced_num_sets": 25, "bogus": True})
