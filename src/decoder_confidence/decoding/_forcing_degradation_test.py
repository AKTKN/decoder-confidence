from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import stim

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._decoder_adapter import (
    RelayBpDecoderAdapter,
    _build_relay_bp_adapter,
    _clip_priors,
    _dem_to_matrices_with_edge_priors,
    _weights_from_priors,
)
from decoder_confidence.decoding._linearize_logicalgap import (
    _logical_from_correction,
)
from decoder_confidence.decoding._lsd_cluster_metric import _compute_cluster_llr
from decoder_confidence.execution.models import DecoderBase, DecoderFactory

FORCING_DEGRADATION_METRIC = "forcing_degradation_test"


def _append_observable_constraints(check_matrix: Any, observables_matrix: Any) -> Any:
    """Append all logical-observable rows to a parity-check matrix.

    Parameters
    ----------
    check_matrix:
        Detector parity-check matrix with shape ``(num_detectors, num_errors)``.
        The matrix may be dense or any SciPy sparse matrix.
    observables_matrix:
        Observable matrix with shape ``(num_observables, num_errors)``.  Each
        row computes one logical-class bit from a correction vector.

    Returns
    -------
    Any
        Matrix with shape ``(num_detectors + num_observables, num_errors)``.

    Raises
    ------
    ValueError
        If the two matrices have incompatible numbers of columns.
    """
    if int(check_matrix.shape[1]) != int(observables_matrix.shape[1]):
        raise ValueError(
            "check_matrix and observables_matrix must have the same number of columns "
            f"({check_matrix.shape[1]} != {observables_matrix.shape[1]})"
        )
    if int(observables_matrix.shape[0]) == 0:
        return check_matrix

    try:
        from scipy import sparse
    except ImportError:
        sparse = None

    if sparse is not None and (
        sparse.issparse(check_matrix) or sparse.issparse(observables_matrix)
    ):
        return sparse.vstack([check_matrix, observables_matrix], format="csc")

    return np.vstack(
        [
            np.asarray(check_matrix, dtype=np.uint8),
            np.asarray(observables_matrix, dtype=np.uint8),
        ]
    )


def _forced_syndrome_for_logical_class(
    syndrome: np.ndarray,
    logical_class: np.ndarray,
) -> np.ndarray:
    """Return the syndrome augmented with target logical-class parity bits.

    The appended rows are the observable matrix rows.  Setting their syndrome
    bits to the baseline logical class forces the second decode to search in
    that same logical class.
    """
    return np.concatenate(
        [
            np.asarray(syndrome, dtype=np.uint8),
            np.asarray(logical_class, dtype=np.uint8),
        ]
    )


@dataclass(frozen=True)
class StageDecode:
    """Single-stage forcing-degradation decode output."""

    correction: np.ndarray
    logical_class: np.ndarray
    weight: float
    metric: float | int
    converged: bool = True


class ForcingStageRunner(Protocol):
    """Decoder-specific operations required by the two-stage simulation."""

    metric_name: str

    @property
    def observables_matrix(self) -> Any:
        ...

    def decode_baseline(self, syndrome: np.ndarray) -> StageDecode:
        ...

    def decode_forced(
        self,
        syndrome: np.ndarray,
        baseline_logical_class: np.ndarray,
    ) -> StageDecode:
        ...

    def reset(self) -> None:
        ...


@dataclass
class BpLsdForcingStageRunner:
    """BP-LSD stage runner that records ``cluster_llr`` for both stages.

    The wrapped BP-LSD decoder is always constructed with
    ``always_run_lsd=True`` and ``set_do_stats(True)`` so that LSD cluster
    statistics exist even for shots where BP converges before LSD fallback.
    """

    check_matrix: Any
    observables_matrix: Any
    priors: np.ndarray
    decoder_options: Mapping[str, Any]
    alpha: float

    metric_name: str = "cluster_llr"

    def __post_init__(self) -> None:
        self._base_check_matrix = self.check_matrix
        self._base_priors = _clip_priors(self.priors)
        self._weights = _weights_from_priors(self._base_priors)
        self._w_e = np.where(
            self._base_priors > 0.0,
            np.log((1.0 - self._base_priors) / self._base_priors),
            np.inf,
        )
        self._forced_check_matrix = _append_observable_constraints(
            self._base_check_matrix, self.observables_matrix
        )
        self._decoder = self._build_decoder(self._base_check_matrix)

    def _build_decoder(self, check_matrix: Any) -> Any:
        try:
            from ldpc.bplsd_decoder import BpLsdDecoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ldpc is required for BP-LSD forcing_degradation_test") from exc

        options = dict(self.decoder_options)
        options["always_run_lsd"] = True
        options.setdefault("input_vector_type", "syndrome")
        decoder = BpLsdDecoder(
            check_matrix,
            error_channel=list(self._base_priors),
            **options,
        )
        decoder.set_do_stats(True)
        return decoder

    def _decode_with_current_matrix(self, syndrome: np.ndarray) -> StageDecode:
        correction = np.asarray(
            self._decoder.decode(np.asarray(syndrome, dtype=np.uint8)),
            dtype=np.bool_,
        )
        stats = self._decoder.statistics or {}
        metric = _compute_cluster_llr(
            stats.get("individual_cluster_stats", {}),
            self._w_e,
            self.alpha,
        )
        logical = _logical_from_correction(self.observables_matrix, correction)
        weight = float(self._weights @ correction.astype(int))
        return StageDecode(
            correction=correction,
            logical_class=logical.astype(np.bool_),
            weight=weight,
            metric=float(metric),
            converged=True,
        )

    def decode_baseline(self, syndrome: np.ndarray) -> StageDecode:
        self._decoder = self._build_decoder(self._base_check_matrix)
        return self._decode_with_current_matrix(syndrome)

    def decode_forced(
        self,
        syndrome: np.ndarray,
        baseline_logical_class: np.ndarray,
    ) -> StageDecode:
        self._decoder = self._build_decoder(self._forced_check_matrix)
        forced_syndrome = _forced_syndrome_for_logical_class(
            syndrome, baseline_logical_class
        )
        return self._decode_with_current_matrix(forced_syndrome)

    def reset(self) -> None:
        self._decoder = self._build_decoder(self._base_check_matrix)


@dataclass
class RelayBpForcingStageRunner:
    """Relay-BP stage runner that records detailed decoder iteration counts."""

    adapter: RelayBpDecoderAdapter

    metric_name: str = "iteration"

    def __post_init__(self) -> None:
        self._base_check_matrix = self.adapter.check_matrix
        self._base_priors = _clip_priors(self.adapter.priors)
        self._observables = self.adapter.observables_matrix
        self._weights = _weights_from_priors(self._base_priors)
        self._forced_check_matrix = _append_observable_constraints(
            self._base_check_matrix, self._observables
        )

    @property
    def observables_matrix(self) -> Any:
        return self._observables

    def _decode_current(self, syndrome: np.ndarray) -> StageDecode:
        result = self.adapter.decode_detailed_single(np.asarray(syndrome, dtype=np.uint8))
        correction = np.asarray(result.decoding, dtype=np.bool_)
        logical = _logical_from_correction(self._observables, correction)
        weight = float(self._weights @ correction.astype(int))
        return StageDecode(
            correction=correction,
            logical_class=logical.astype(np.bool_),
            weight=weight,
            metric=int(result.iterations),
            converged=bool(result.success),
        )

    def decode_baseline(self, syndrome: np.ndarray) -> StageDecode:
        self.adapter.set_check_matrix(self._base_check_matrix)
        self.adapter.set_priors(self._base_priors.copy())
        return self._decode_current(syndrome)

    def decode_forced(
        self,
        syndrome: np.ndarray,
        baseline_logical_class: np.ndarray,
    ) -> StageDecode:
        self.adapter.set_check_matrix(self._forced_check_matrix)
        self.adapter.set_priors(self._base_priors.copy())
        forced_syndrome = _forced_syndrome_for_logical_class(
            syndrome, baseline_logical_class
        )
        return self._decode_current(forced_syndrome)

    def reset(self) -> None:
        self.adapter.set_check_matrix(self._base_check_matrix)
        self.adapter.set_priors(self._base_priors.copy())


@dataclass
class ForcingDegradationTestDecoder(DecoderBase):
    """Run baseline and same-logical-class forced decodes for each syndrome.

    Parameters
    ----------
    runner:
        Decoder-specific stage runner.  It owns any mutable decoder state and
        returns corrections, logical classes, weights, and per-stage metrics.

    Returns
    -------
    DecodingResult
        ``predictions`` is the baseline logical-class matrix with shape
        ``(num_shots, num_observables)``.  ``metrics`` contains
        ``baseline_weight``, ``forced_weight``, and the stage-prefixed
        decoder-specific metric columns.  The execution worker derives
        ``is_logical_error`` from these baseline predictions and sampled
        observables.

    Raises
    ------
    ValueError
        If syndromes are not one- or two-dimensional.
    RuntimeError
        If a forced decode returns a correction outside the baseline logical
        class imposed by the observable-matrix constraint.
    """

    runner: ForcingStageRunner

    def decode(self, syndromes: np.ndarray) -> DecodingResult:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        if syndromes.ndim != 2:
            raise ValueError("syndromes must be a 1D or 2D array")

        num_shots = syndromes.shape[0]
        num_obs = int(self.runner.observables_matrix.shape[0])
        metric_name = self.runner.metric_name

        predictions = np.zeros((num_shots, num_obs), dtype=np.bool_)
        force_logical_error = np.zeros(num_shots, dtype=np.bool_)
        baseline_weight = np.zeros(num_shots, dtype=np.float64)
        forced_weight = np.zeros(num_shots, dtype=np.float64)
        if metric_name == "iteration":
            baseline_metric: np.ndarray = np.zeros(num_shots, dtype=np.int64)
            forced_metric: np.ndarray = np.zeros(num_shots, dtype=np.int64)
        else:
            baseline_metric = np.zeros(num_shots, dtype=np.float64)
            forced_metric = np.zeros(num_shots, dtype=np.float64)

        try:
            for shot in range(num_shots):
                syndrome = syndromes[shot]
                baseline = self.runner.decode_baseline(syndrome)
                forced = self.runner.decode_forced(syndrome, baseline.logical_class)

                both_converged = baseline.converged and forced.converged
                if not both_converged:
                    force_logical_error[shot] = True
                elif not np.array_equal(forced.logical_class, baseline.logical_class):
                    baseline_bits = baseline.logical_class.astype(int).tolist()
                    forced_bits = forced.logical_class.astype(int).tolist()
                    raise RuntimeError(
                        "Forced run logical class does not match baseline logical class "
                        f"for shot index {shot}: baseline={baseline_bits} "
                        f"forced={forced_bits}"
                    )

                predictions[shot] = baseline.logical_class.astype(np.bool_)
                baseline_weight[shot] = baseline.weight
                forced_weight[shot] = forced.weight
                baseline_metric[shot] = baseline.metric
                forced_metric[shot] = forced.metric
        finally:
            self.runner.reset()

        return DecodingResult(
            predictions=predictions,
            metrics={
                "baseline_weight": baseline_weight,
                "forced_weight": forced_weight,
                f"baseline_{metric_name}": baseline_metric,
                f"forced_{metric_name}": forced_metric,
                "__is_logical_error": force_logical_error,
            },
        )


def _parse_alpha(metric_options: Mapping[str, Any]) -> float:
    raw = metric_options.get("alpha", metric_options.get("b", 2.0))
    alpha = float(raw) if str(raw).lower() != "inf" else np.inf
    if alpha <= 0.0:
        raise ValueError(f"metric_options.alpha must be > 0, got {raw}")
    return alpha


@dataclass(frozen=True)
class _ForcingDegradationFactory:
    dem_path: Path
    decoder_name: str
    decoder_options: Mapping[str, Any]
    metric_options: Mapping[str, Any]

    def __call__(
        self, dem: stim.DetectorErrorModel | None = None
    ) -> ForcingDegradationTestDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

        if self.decoder_name in {"BP-LSD", "BPLSD"}:
            matrices = _dem_to_matrices_with_edge_priors(
                dem, allow_undecomposed_hyperedges=True
            )
            runner = BpLsdForcingStageRunner(
                check_matrix=matrices.check_matrix,
                observables_matrix=matrices.observables_matrix,
                priors=_clip_priors(matrices.priors),
                decoder_options=dict(self.decoder_options),
                alpha=_parse_alpha(self.metric_options),
            )
            return ForcingDegradationTestDecoder(runner=runner)

        if self.decoder_name == "RELAY-BP":
            adapter = _build_relay_bp_adapter(dem, self.decoder_options)
            return ForcingDegradationTestDecoder(
                runner=RelayBpForcingStageRunner(adapter=adapter)
            )

        raise ValueError(
            "forcing_degradation_test supports only BP-LSD and RELAY-BP, "
            f"got {self.decoder_name}"
        )


def make_forcing_degradation_test_factory(
    dem_path: Path,
    decoder_name: str,
    decoder_options: Mapping[str, Any],
    metric_options: Mapping[str, Any],
) -> DecoderFactory:
    """Create a decoder factory for the ``forcing_degradation_test`` workflow."""
    normalized = decoder_name.strip().upper().replace("_", "-")
    if normalized not in {"BP-LSD", "BPLSD", "RELAY-BP"}:
        raise ValueError(
            "forcing_degradation_test supports only BP-LSD and RELAY-BP, "
            f"got {decoder_name}"
        )
    return _ForcingDegradationFactory(
        dem_path=dem_path,
        decoder_name=normalized,
        decoder_options=dict(decoder_options),
        metric_options=dict(metric_options),
    )
