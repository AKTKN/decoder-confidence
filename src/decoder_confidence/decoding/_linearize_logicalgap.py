from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import stim

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._decoder_adapter import (
    DecoderAdapter,
    build_decoder_adapter,
    _clip_priors,
    _weights_from_priors,
)
from decoder_confidence.execution.models import DecoderBase, DecoderFactory


@dataclass(frozen=True)
class LinearizeLogicalGapOptions:
    """Options for the linearize_logicalgap metric. Extensible for future additions."""


def _parse_linearize_options(metric_options: Mapping[str, Any]) -> LinearizeLogicalGapOptions:
    return LinearizeLogicalGapOptions()


def _get_obs_row(observables: Any, i: int) -> np.ndarray:
    """Return the i-th row of the observables matrix as a 1-D int array."""
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    if sparse is not None and sparse.issparse(observables):
        return np.asarray(observables[i, :].todense(), dtype=int).ravel()
    return np.asarray(observables, dtype=int)[i, :]


def _append_row(check_matrix: Any, row: np.ndarray) -> Any:
    """Return a new matrix with *row* appended as the last row."""
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    if sparse is not None and sparse.issparse(check_matrix):
        row_sparse = sparse.csr_matrix(row.reshape(1, -1))
        return sparse.vstack([check_matrix, row_sparse], format="csc")
    return np.vstack([np.asarray(check_matrix), row.reshape(1, -1)])


def _logical_from_correction(observables: Any, correction: np.ndarray) -> np.ndarray:
    """Compute (observables @ correction) % 2 for sparse or dense observables."""
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    vec = np.asarray(correction, dtype=int)
    if sparse is not None and sparse.issparse(observables):
        return np.asarray((observables @ vec) % 2).ravel()
    return (np.asarray(observables, dtype=int) @ vec) % 2


@dataclass
class LinearizeLogicalGapDecoder(DecoderBase):
    """Two-stage decoder that estimates the logical gap via augmented check constraints.

    Stage 1: decode normally to obtain the best correction c1 in logical class l1.
    Stage 2: for each observable i, decode with an extra parity constraint that forces
    the correction to flip bit i of the logical class. The gap is the difference between
    the minimum second-stage weight and the first-stage weight.
    """

    adapter: DecoderAdapter
    options: LinearizeLogicalGapOptions

    def __post_init__(self) -> None:
        self._base_priors: np.ndarray = _clip_priors(self.adapter.priors)
        self._base_check_matrix: Any = self.adapter.check_matrix
        self._observables: Any = self.adapter.observables_matrix
        self._num_errors: int = self.adapter.num_errors
        self._weights: np.ndarray = _weights_from_priors(self._base_priors)

    def decode(self, syndromes: np.ndarray) -> DecodingResult:
        syndromes = np.asarray(syndromes, dtype=int)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        if syndromes.ndim != 2:
            raise ValueError("syndromes must be 1D or 2D array")

        num_shots = syndromes.shape[0]
        num_obs = int(self._observables.shape[0])

        predictions = np.zeros((num_shots, num_obs), dtype=np.bool_)
        gap = np.full((num_shots,), np.nan, dtype=float)

        for shot in range(num_shots):
            syndrome = syndromes[shot]

            # --- Stage 1: decode with the original check matrix ---
            self.adapter.set_priors(self._base_priors.copy())
            self.adapter.set_check_matrix(self._base_check_matrix)
            c1 = np.asarray(self.adapter.decode(syndrome), dtype=np.bool_)

            l1 = _logical_from_correction(self._observables, c1)
            w1 = float(self._weights @ c1.astype(int))
            predictions[shot] = l1.astype(np.bool_)

            if num_obs == 0:
                continue

            # --- Stage 2: one decode per observable, forcing it to flip ---
            best_w2 = np.inf
            for i in range(num_obs):
                obs_row_i = _get_obs_row(self._observables, i)
                aug_check = _append_row(self._base_check_matrix, obs_row_i)
                # Syndrome bit for the new check: (L[i] @ E) mod 2 must equal 1 XOR l1[i]
                target_bit = 1 - int(l1[i])
                aug_syndrome = np.append(syndrome, target_bit)

                self.adapter.set_check_matrix(aug_check)
                self.adapter.set_priors(self._base_priors.copy())
                c2 = np.asarray(self.adapter.decode(aug_syndrome), dtype=np.bool_)

                w2_i = float(self._weights @ c2.astype(int))
                if w2_i < best_w2:
                    best_w2 = w2_i

            # Restore original state for the next shot
            self.adapter.set_check_matrix(self._base_check_matrix)
            self.adapter.set_priors(self._base_priors.copy())

            gap[shot] = best_w2 - w1

        return DecodingResult(
            predictions=predictions,
            metrics={"linearize_logicalgap": gap},
        )


@dataclass(frozen=True)
class _LinearizeLogicalGapFactory:
    dem_path: Path
    base_decoder: str
    decoder_options: Mapping[str, Any]
    metric_options: Mapping[str, Any]
    use_edge_matrices: bool = False

    def __call__(
        self, dem: stim.DetectorErrorModel | None = None
    ) -> LinearizeLogicalGapDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

        adapter = build_decoder_adapter(
            self.base_decoder, dem, self.decoder_options,
            use_edge_matrices=self.use_edge_matrices,
        )
        options = _parse_linearize_options(self.metric_options)

        return LinearizeLogicalGapDecoder(adapter=adapter, options=options)


def make_linearize_logicalgap_factory(
    dem_path: Path,
    base_decoder: str,
    decoder_options: Mapping[str, Any],
    metric_options: Mapping[str, Any],
    use_edge_matrices: bool = False,
) -> DecoderFactory:
    return _LinearizeLogicalGapFactory(
        dem_path=dem_path,
        base_decoder=base_decoder,
        decoder_options=dict(decoder_options),
        metric_options=dict(metric_options),
        use_edge_matrices=use_edge_matrices,
    )
