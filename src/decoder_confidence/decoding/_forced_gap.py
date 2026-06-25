from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import stim

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._constraints import (
    ConstrainedDecodeOptions,
    build_constrained_system,
)
from decoder_confidence.decoding._decoder_adapter import (
    DecoderAdapter,
    BpLsdDecoderAdapter,
    RelayBpDecoderAdapter,
    build_decoder_adapter,
    _clip_priors,
    _weights_from_priors,
)
from decoder_confidence.decoding._lsd_cluster_metric import _compute_cluster_llr
from decoder_confidence.decoding._linearize_logicalgap import (
    _get_obs_row,
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
    random_split: bool = False
    n_splits: int = 3
    split_seed: int = 0
    split_balanced: bool = False
    cluster_llr_alpha: float = 2.0

    @property
    def constrained_decode_options(self) -> ConstrainedDecodeOptions:
        return ConstrainedDecodeOptions(
            random_split=self.random_split,
            n_splits=self.n_splits,
            split_seed=self.split_seed,
            split_balanced=self.split_balanced,
        )


def _parse_forced_gap_ml_options(metric_options: Mapping[str, Any]) -> ForcedGapMLOptions:
    options = dict(metric_options)
    allowed = {
        "alpha",
        "cluster_llr_alpha",
        "get_all_failure_rate",
        "get_detail_stat",
        "random_split",
        "n_splits",
        "split_seed",
        "split_balanced",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            "Unsupported forced_gap_ml metric option(s): "
            + ", ".join(unknown)
            + f". Supported options: {', '.join(sorted(allowed))}"
        )

    get_all_failure_rate = bool(options.get("get_all_failure_rate", False))
    get_detail_stat = bool(options.get("get_detail_stat", False))
    alpha_raw = options.get("alpha", options.get("cluster_llr_alpha", 2.0))
    alpha = float(alpha_raw) if str(alpha_raw).lower() != "inf" else np.inf
    return ForcedGapMLOptions(
        get_all_failure_rate=get_all_failure_rate,
        get_detail_stat=get_detail_stat,
        random_split=bool(options.get("random_split", False)),
        n_splits=int(options.get("n_splits", 3)),
        split_seed=int(options.get("split_seed", 0)),
        split_balanced=bool(options.get("split_balanced", False)),
        cluster_llr_alpha=alpha,
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
        self._partition_cache: dict[int, Any] = {}
        self._decoder_stat_kind: str | None = None
        if self.options.get_detail_stat:
            if isinstance(self.adapter, BpLsdDecoderAdapter):
                self.adapter.enable_lsd_statistics()
                self._decoder_stat_kind = "cluster_llr"
            elif isinstance(self.adapter, RelayBpDecoderAdapter):
                self._decoder_stat_kind = "iteration"
            else:
                warnings.warn(
                    "decoder_stat.parquet is only supported for BP-LSD and Relay-BP; "
                    f"got {self.adapter.__class__.__name__}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _decode_with_decoder_stat(self, syndrome: np.ndarray) -> tuple[np.ndarray, float | None]:
        if self._decoder_stat_kind == "iteration":
            assert isinstance(self.adapter, RelayBpDecoderAdapter)
            result = self.adapter.decode_detailed_single(syndrome)
            stat = float(result.iterations) if bool(result.success) else np.nan
            return np.asarray(result.decoding, dtype=np.bool_), stat

        correction = np.asarray(self.adapter.decode(syndrome), dtype=np.bool_)
        if self._decoder_stat_kind == "cluster_llr":
            assert isinstance(self.adapter, BpLsdDecoderAdapter)
            stats = self.adapter.lsd_statistics
            priors = np.asarray(self.adapter.priors, dtype=float)
            w_e = _weights_from_priors(priors)
            return (
                correction,
                _compute_cluster_llr(
                    stats.get("individual_cluster_stats", {}),
                    w_e,
                    self.options.cluster_llr_alpha,
                ),
            )
        return correction, None

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

            baseline_stat = np.full((num_shots,), np.nan, dtype=float)
            forced_stat = np.full((num_shots,), np.nan, dtype=float)
            forced_2ndbest_stat = np.full((num_shots,), np.nan, dtype=float)
            forced_mean_stat = np.full((num_shots,), np.nan, dtype=float)
            forced_std_stat = np.full((num_shots,), np.nan, dtype=float)
            forced_max_stat = np.full((num_shots,), np.nan, dtype=float)

        for shot in range(num_shots):
            syndrome = syndromes[shot]

            # --- Stage 1: decode with the original check matrix ---
            self.adapter.set_check_matrix_and_priors(
                self._base_check_matrix,
                self._base_priors.copy(),
            )
            c1, stat1 = self._decode_with_decoder_stat(syndrome)

            l1 = _logical_from_correction(self._observables, c1)
            w1 = float(self._weights @ c1.astype(int))
            obs_shot = true_obs_arr[shot] if true_obs_arr is not None else None
            if self.options.get_detail_stat:
                stage1_weight[shot] = w1
                stage1_obs_flip[shot] = _is_obs_flip(l1, obs_shot)
                if stat1 is not None:
                    baseline_stat[shot] = stat1

            # Collect all solutions: list of (weight, logical_class)
            all_solutions: list[tuple[float, np.ndarray]] = [(w1, l1)]
            stage2_solutions: list[tuple[float, np.ndarray, float | None]] = []
            stage2_logicals: list[np.ndarray] = []
            stage2_stats: list[float] = []

            if num_obs > 0:
                # --- Stage 2: one decode per observable, forcing it to flip ---
                for i in range(num_obs):
                    obs_row_i = _get_obs_row(self._observables, i)
                    target_bit = 1 - int(l1[i])
                    constrained = build_constrained_system(
                        self._base_check_matrix,
                        syndrome,
                        self._base_priors,
                        obs_row_i,
                        target_bit,
                        self.options.constrained_decode_options,
                        self._partition_cache,
                        i,
                    )

                    self.adapter.set_check_matrix_and_priors(
                        constrained.check_matrix,
                        constrained.priors.copy(),
                    )
                    c2_full, stat2 = self._decode_with_decoder_stat(constrained.syndrome)
                    c2 = c2_full[: constrained.physical_cols]

                    l2 = _logical_from_correction(self._observables, c2)
                    w2_i = float(self._weights @ c2.astype(int))
                    if stat2 is not None:
                        stage2_stats.append(float(stat2))
                    stage2_logicals.append(l2)
                    stage2_solutions.append((w2_i, l2, stat2))
                    all_solutions.append((w2_i, l2))

                # Restore original state for the next shot
                self.adapter.set_check_matrix_and_priors(
                    self._base_check_matrix,
                    self._base_priors.copy(),
                )

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
                best_w2, best_l2, best_stat2 = stage2_solutions[0]
                stage2_weight[shot] = best_w2
                stage2_obs_flip[shot] = _is_obs_flip(best_l2, obs_shot)
                if best_stat2 is not None:
                    forced_stat[shot] = best_stat2
                if len(stage2_solutions) >= 2:
                    second_w2, second_l2, second_stat2 = stage2_solutions[1]
                    stage2_2ndbest_weight[shot] = second_w2
                    stage2_2ndbest_obs_flip[shot] = _is_obs_flip(second_l2, obs_shot)
                    if second_stat2 is not None:
                        forced_2ndbest_stat[shot] = second_stat2
                if stage2_stats:
                    stat_arr = np.asarray(stage2_stats, dtype=float)
                    if np.isfinite(stat_arr).any():
                        forced_mean_stat[shot] = float(np.nanmean(stat_arr))
                        forced_std_stat[shot] = float(np.nanstd(stat_arr))
                        forced_max_stat[shot] = float(np.nanmax(stat_arr))

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
        detail_stats: dict[str, Any] = {}
        decoder_stats: dict[str, Any] = {}
        if self.options.get_detail_stat:
            detail_stats = {
                "stage1_weight": stage1_weight,
                "stage1_obs_flip": stage1_obs_flip,
                "stage2_weight": stage2_weight,
                "stage2_obs_flip": stage2_obs_flip,
                "stage2_2ndbest_weight": stage2_2ndbest_weight,
                "stage2_2ndbest_obs_flip": stage2_2ndbest_obs_flip,
            }
            if self._decoder_stat_kind == "cluster_llr":
                decoder_stats = {
                    "baseline_cluster_llr": baseline_stat,
                    "forced_cluster_llr": forced_stat,
                    "forced_2nd_best_cluster_llr": forced_2ndbest_stat,
                    "forced_mean_cluster_llr": forced_mean_stat,
                    "forced_std_cluster_llr": forced_std_stat,
                    "forced_max_cluster_llr": forced_max_stat,
                }
            elif self._decoder_stat_kind == "iteration":
                decoder_stats = {
                    "baseline_iteration": baseline_stat,
                    "forced_iteration": forced_stat,
                    "forced_2nd_best_iteration": forced_2ndbest_stat,
                    "forced_mean_iteration": forced_mean_stat,
                    "forced_std_iteration": forced_std_stat,
                    "forced_max_iteration": forced_max_stat,
                }

        return DecodingResult(
            predictions=predictions,
            metrics=metrics,
            detail_stats=detail_stats,
            decoder_stats=decoder_stats,
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
