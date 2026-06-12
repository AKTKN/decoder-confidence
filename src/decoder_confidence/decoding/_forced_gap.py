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
    _is_obs_flip,
    _logical_from_correction,
    _normalize_true_obs,
)
from decoder_confidence.execution.models import DecoderBase, DecoderFactory

# Per-shot case labels produced when get_all_failure_rate=True.
# These classify the outcome of the two-stage decoding relative to the true
# observables for each shot.
#
# K = num_obs  (number of logical observables; Stage 2 runs K decodes)
#
#   0: All K+1 solutions (1 from Stage 1, K from Stage 2) are logical errors.
#   1: Stage 1 solution IS a logical error; at least one Stage 2 solution is NOT
#      a logical error, AND that non-error solution was adopted as the final answer
#      (it had the smallest weight).
#   2: Stage 1 solution IS a logical error; at least one Stage 2 solution is NOT
#      a logical error, BUT that non-error solution was NOT adopted as the final
#      answer (it was not the minimum-weight solution overall).
#   3: Stage 1 solution is NOT a logical error, but a Stage 2 solution with
#      strictly smaller weight was found and adopted instead, overriding Stage 1.
#  -1: Stage 1 solution is NOT a logical error and was adopted as the final answer
#      (normal success; does not fall into any of cases 0-3).


@dataclass(frozen=True)
class ForcedGapMLOptions:
    """Options for the forced_gap_ml metric."""

    get_all_failure_rate: bool = False
    get_detail_stat: bool = False


def _parse_forced_gap_ml_options(metric_options: Mapping[str, Any]) -> ForcedGapMLOptions:
    get_all_failure_rate = bool(metric_options.get("get_all_failure_rate", False))
    get_detail_stat = bool(metric_options.get("get_detail_stat", False))
    return ForcedGapMLOptions(
        get_all_failure_rate=get_all_failure_rate,
        get_detail_stat=get_detail_stat,
    )


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

    def decode(
        self,
        syndromes: np.ndarray,
        *,
        true_obs: np.ndarray | None = None,
    ) -> DecodingResult:
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
        true_obs_arr = _normalize_true_obs(true_obs, num_shots, num_obs)

        compute_cases = self.options.get_all_failure_rate and true_obs_arr is not None
        if compute_cases:
            # case_labels: -1 = normal success (stage1 correct and adopted)
            case_labels = np.full(num_shots, -1, dtype=np.int8)
        if self.options.get_detail_stat:
            if num_obs > 0 and true_obs_arr is None:
                raise ValueError("true_obs is required when get_detail_stat=True")
            stage1_weight = np.full((num_shots,), np.nan, dtype=float)
            stage1_obs_flip = np.zeros((num_shots,), dtype=np.bool_)
            stage2_weight = np.full((num_shots,), np.nan, dtype=float)
            stage2_obs_flip = np.zeros((num_shots,), dtype=np.bool_)
            stage2_2ndbest_weight = np.full((num_shots,), np.nan, dtype=float)
            stage2_2ndbest_obs_flip = np.zeros((num_shots,), dtype=np.bool_)

        for shot in range(num_shots):
            syndrome = syndromes[shot]

            # --- Stage 1: decode with the original check matrix ---
            self.adapter.set_priors(self._base_priors.copy())
            self.adapter.set_check_matrix(self._base_check_matrix)
            c1 = np.asarray(self.adapter.decode(syndrome), dtype=np.bool_)

            l1 = _logical_from_correction(self._observables, c1)
            w1 = float(self._weights @ c1.astype(int))
            obs_shot = true_obs_arr[shot] if true_obs_arr is not None else None
            if self.options.get_detail_stat:
                stage1_weight[shot] = w1
                stage1_obs_flip[shot] = _is_obs_flip(l1, obs_shot)

            # Collect all solutions: list of (weight, logical_class)
            all_solutions: list[tuple[float, np.ndarray]] = [(w1, l1)]
            stage2_solutions: list[tuple[float, np.ndarray]] = []
            stage2_logicals: list[np.ndarray] = []

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
                    stage2_logicals.append(l2)
                    stage2_solutions.append((w2_i, l2))
                    all_solutions.append((w2_i, l2))

                # Restore original state for the next shot
                self.adapter.set_check_matrix(self._base_check_matrix)
                self.adapter.set_priors(self._base_priors.copy())

            # Sort by weight; stable sort preserves Stage-1 priority on ties
            all_solutions.sort(key=lambda x: x[0])

            ml_weight, ml_logical = all_solutions[0]
            predictions[shot] = ml_logical.astype(np.bool_)

            # Gap: min weight among solutions in a different logical class - ml_weight
            # all_solutions is sorted, so iterate to find the first different-class solution
            best_diff_weight = np.inf
            for w, l in all_solutions[1:]:
                if not np.array_equal(l, ml_logical):
                    best_diff_weight = w
                    break
            gap[shot] = (best_diff_weight - ml_weight) if np.isfinite(best_diff_weight) else np.nan

            diff = np.asarray(l1, dtype=int) ^ np.asarray(ml_logical, dtype=int)
            obs_flip_idx.append(list(np.where(diff)[0]))

            if self.options.get_detail_stat and stage2_solutions:
                stage2_solutions.sort(key=lambda x: x[0])
                best_w2, best_l2 = stage2_solutions[0]
                stage2_weight[shot] = best_w2
                stage2_obs_flip[shot] = _is_obs_flip(best_l2, obs_shot)
                if len(stage2_solutions) >= 2:
                    second_w2, second_l2 = stage2_solutions[1]
                    stage2_2ndbest_weight[shot] = second_w2
                    stage2_2ndbest_obs_flip[shot] = _is_obs_flip(second_l2, obs_shot)

            if compute_cases:
                assert obs_shot is not None
                stage1_is_error = bool(np.any(l1 != obs_shot))
                final_is_error = bool(np.any(ml_logical != obs_shot))
                stage1_adopted = np.array_equal(l1, ml_logical)
                any_stage2_correct = any(
                    not np.any(l2 != obs_shot) for l2 in stage2_logicals
                )

                if stage1_is_error:
                    if not any_stage2_correct:
                        case_labels[shot] = 0
                    elif not final_is_error:
                        case_labels[shot] = 1
                    else:
                        case_labels[shot] = 2
                elif not stage1_adopted:
                    case_labels[shot] = 3
                # else -1 (default): stage1 correct and adopted

        metrics: dict[str, Any] = {"forced_gap_ml": gap}
        if compute_cases:
            metrics["forced_gap_ml_case"] = case_labels
        if self.options.get_detail_stat:
            metrics.update(
                {
                    "stage1_weight": stage1_weight,
                    "stage1_obs_flip": stage1_obs_flip,
                    "stage2_weight": stage2_weight,
                    "stage2_obs_flip": stage2_obs_flip,
                    "stage2_2ndbest_weight": stage2_2ndbest_weight,
                    "stage2_2ndbest_obs_flip": stage2_2ndbest_obs_flip,
                }
            )

        return DecodingResult(
            predictions=predictions,
            metrics=metrics,
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
