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
from decoder_confidence.decoding._linearize_logicalgap import (
    _get_obs_row,
    _append_row,
    _logical_from_correction,
)
from decoder_confidence.execution.models import DecoderBase, DecoderFactory


@dataclass(frozen=True)
class ForcedGapMLOptions:
    """Options for the forced_gap_ml metric."""


def _parse_forced_gap_ml_options(metric_options: Mapping[str, Any]) -> ForcedGapMLOptions:
    return ForcedGapMLOptions()


@dataclass
class ForcedGapMLDecoder(DecoderBase):
    """Two-stage ML decoder that estimates confidence via the gap between the two
    lowest-weight solutions across all logical classes.

    Stage 1: decode normally → correction c1, weight w1, logical class l1.
    Stage 2: for each observable i, decode with an extra parity constraint that forces
    the correction into the complementary logical class for bit i.

    Among the 1+k total solutions (1 from Stage 1, k from Stage 2) the decoder
    selects the minimum-weight solution as the final prediction and reports
        gap = (2nd-smallest weight) - (smallest weight)
    as the confidence metric.
    """

    adapter: DecoderAdapter
    options: ForcedGapMLOptions

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
        obs_flip_idx: list[list[int]] = []

        for shot in range(num_shots):
            syndrome = syndromes[shot]

            # --- Stage 1: decode with the original check matrix ---
            self.adapter.set_priors(self._base_priors.copy())
            self.adapter.set_check_matrix(self._base_check_matrix)
            c1 = np.asarray(self.adapter.decode(syndrome), dtype=np.bool_)

            l1 = _logical_from_correction(self._observables, c1)
            w1 = float(self._weights @ c1.astype(int))

            # Collect all solutions: list of (weight, logical_class)
            all_solutions: list[tuple[float, np.ndarray]] = [(w1, l1)]

            if num_obs > 0:
                # --- Stage 2: one decode per observable, forcing it to flip ---
                for i in range(num_obs):
                    obs_row_i = _get_obs_row(self._observables, i)
                    aug_check = _append_row(self._base_check_matrix, obs_row_i)
                    target_bit = 1 - int(l1[i])
                    aug_syndrome = np.append(syndrome, target_bit)

                    self.adapter.set_check_matrix(aug_check)
                    self.adapter.set_priors(self._base_priors.copy())
                    c2 = np.asarray(self.adapter.decode(aug_syndrome), dtype=np.bool_)

                    l2 = _logical_from_correction(self._observables, c2)
                    w2_i = float(self._weights @ c2.astype(int))
                    all_solutions.append((w2_i, l2))

                # Restore original state for the next shot
                self.adapter.set_check_matrix(self._base_check_matrix)
                self.adapter.set_priors(self._base_priors.copy())

            # Sort by weight; stable sort preserves Stage-1 priority on ties
            all_solutions.sort(key=lambda x: x[0])

            ml_weight, ml_logical = all_solutions[0]
            predictions[shot] = ml_logical.astype(np.bool_)

            if len(all_solutions) >= 2:
                gap[shot] = all_solutions[1][0] - ml_weight
            else:
                gap[shot] = np.nan

            diff = np.asarray(l1, dtype=int) ^ np.asarray(ml_logical, dtype=int)
            obs_flip_idx.append(list(np.where(diff)[0]))

        return DecodingResult(
            predictions=predictions,
            metrics={"forced_gap_ml": gap},
            obs_flip_idx=obs_flip_idx,
        )


@dataclass(frozen=True)
class _ForcedGapMLFactory:
    dem_path: Path
    base_decoder: str
    decoder_options: Mapping[str, Any]
    metric_options: Mapping[str, Any]

    def __call__(
        self, dem: stim.DetectorErrorModel | None = None
    ) -> ForcedGapMLDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

        adapter = build_decoder_adapter(self.base_decoder, dem, self.decoder_options)
        options = _parse_forced_gap_ml_options(self.metric_options)

        return ForcedGapMLDecoder(adapter=adapter, options=options)


def make_forced_gap_ml_factory(
    dem_path: Path,
    base_decoder: str,
    decoder_options: Mapping[str, Any],
    metric_options: Mapping[str, Any],
) -> DecoderFactory:
    return _ForcedGapMLFactory(
        dem_path=dem_path,
        base_decoder=base_decoder,
        decoder_options=dict(decoder_options),
        metric_options=dict(metric_options),
    )
