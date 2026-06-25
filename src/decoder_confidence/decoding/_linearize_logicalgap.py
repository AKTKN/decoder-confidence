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
from decoder_confidence.execution.models import DecoderBase, DecoderFactory


@dataclass(frozen=True)
class LinearizeLogicalGapOptions:
    """Options for the linearize_logicalgap metric."""

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


def _parse_linearize_options(metric_options: Mapping[str, Any]) -> LinearizeLogicalGapOptions:
    options = dict(metric_options)
    allowed = {
        "alpha",
        "cluster_llr_alpha",
        "get_detail_stat",
        "random_split",
        "n_splits",
        "split_seed",
        "split_balanced",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            "Unsupported linearize_logicalgap metric option(s): "
            + ", ".join(unknown)
            + f". Supported options: {', '.join(sorted(allowed))}"
        )

    alpha_raw = options.get("alpha", options.get("cluster_llr_alpha", 2.0))
    alpha = float(alpha_raw) if str(alpha_raw).lower() != "inf" else np.inf
    return LinearizeLogicalGapOptions(
        get_detail_stat=bool(options.get("get_detail_stat", False)),
        random_split=bool(options.get("random_split", False)),
        n_splits=int(options.get("n_splits", 3)),
        split_seed=int(options.get("split_seed", 0)),
        split_balanced=bool(options.get("split_balanced", False)),
        cluster_llr_alpha=alpha,
    )


def _normalize_true_obs(
    true_obs: np.ndarray | None, num_shots: int, num_obs: int
) -> np.ndarray | None:
    if true_obs is None:
        return None
    true_obs_arr = np.asarray(true_obs, dtype=np.bool_)
    if true_obs_arr.ndim == 1 and num_obs == 1:
        true_obs_arr = true_obs_arr.reshape(-1, 1)
    if true_obs_arr.shape != (num_shots, num_obs):
        raise ValueError(
            f"true_obs must have shape ({num_shots}, {num_obs}), got {true_obs_arr.shape}"
        )
    return true_obs_arr


def _is_obs_flip(logical: np.ndarray, true_obs_shot: np.ndarray | None) -> bool:
    if true_obs_shot is None:
        return False
    return bool(np.any(np.asarray(logical, dtype=np.bool_) != true_obs_shot))


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

        if self.options.get_detail_stat:
            if num_obs > 0 and true_obs_arr is None:
                raise ValueError("true_obs is required when get_detail_stat=True")
            stage1_weight = np.full((num_shots,), np.nan, dtype=float)
            stage1_obs_flip = np.zeros((num_shots,), dtype=np.bool_)
            stage2_weight = np.full((num_shots,), np.nan, dtype=float)
            stage2_obs_flip = np.zeros((num_shots,), dtype=np.bool_)

            baseline_stat = np.full((num_shots,), np.nan, dtype=float)
            forced_stat = np.full((num_shots,), np.nan, dtype=float)
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
            predictions[shot] = l1.astype(np.bool_)
            obs_shot = true_obs_arr[shot] if true_obs_arr is not None else None

            if self.options.get_detail_stat:
                stage1_weight[shot] = w1
                stage1_obs_flip[shot] = _is_obs_flip(l1, obs_shot)
                if stat1 is not None:
                    baseline_stat[shot] = stat1

            if num_obs == 0:
                obs_flip_idx.append([])
                continue

            # --- Stage 2: one decode per observable, forcing it to flip ---
            best_w2 = np.inf
            best_l2: np.ndarray | None = None
            best_stat2: float | None = None
            stage2_stats: list[float] = []
            best_flip: list[int] = []
            for i in range(num_obs):
                obs_row_i = _get_obs_row(self._observables, i)
                # Syndrome bit for the new check: (L[i] @ E) mod 2 must equal 1 XOR l1[i]
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
                if w2_i < best_w2:
                    best_w2 = w2_i
                    best_l2 = l2
                    best_stat2 = stat2
                    # Indices where stage-2 logical class differs from stage-1
                    diff = np.asarray(l1, dtype=int) ^ np.asarray(l2, dtype=int)
                    best_flip = list(np.where(diff)[0])

            # Restore original state for the next shot
            self.adapter.set_check_matrix_and_priors(
                self._base_check_matrix,
                self._base_priors.copy(),
            )

            gap[shot] = best_w2 - w1
            obs_flip_idx.append(best_flip)
            if self.options.get_detail_stat:
                stage2_weight[shot] = best_w2
                if best_l2 is not None:
                    stage2_obs_flip[shot] = _is_obs_flip(best_l2, obs_shot)
                if best_stat2 is not None:
                    forced_stat[shot] = best_stat2
                if stage2_stats:
                    stat_arr = np.asarray(stage2_stats, dtype=float)
                    if np.isfinite(stat_arr).any():
                        forced_mean_stat[shot] = float(np.nanmean(stat_arr))
                        forced_std_stat[shot] = float(np.nanstd(stat_arr))
                        forced_max_stat[shot] = float(np.nanmax(stat_arr))

        metrics: dict[str, Any] = {"linearize_logicalgap": gap}
        detail_stats: dict[str, Any] = {}
        decoder_stats: dict[str, Any] = {}
        if self.options.get_detail_stat:
            detail_stats = {
                "stage1_weight": stage1_weight,
                "stage1_obs_flip": stage1_obs_flip,
                "stage2_weight": stage2_weight,
                "stage2_obs_flip": stage2_obs_flip,
            }
            if self._decoder_stat_kind == "cluster_llr":
                decoder_stats = {
                    "baseline_cluster_llr": baseline_stat,
                    "forced_cluster_llr": forced_stat,
                    "forced_mean_cluster_llr": forced_mean_stat,
                    "forced_std_cluster_llr": forced_std_stat,
                    "forced_max_cluster_llr": forced_max_stat,
                }
            elif self._decoder_stat_kind == "iteration":
                decoder_stats = {
                    "baseline_iteration": baseline_stat,
                    "forced_iteration": forced_stat,
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
class _LinearizeLogicalGapFactory:
    dem_path: Path
    base_decoder: str
    decoder_options: Mapping[str, Any]
    metric_options: Mapping[str, Any]

    def __call__(
        self, dem: stim.DetectorErrorModel | None = None
    ) -> LinearizeLogicalGapDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

        adapter = build_decoder_adapter(self.base_decoder, dem, self.decoder_options)
        options = _parse_linearize_options(self.metric_options)

        return LinearizeLogicalGapDecoder(adapter=adapter, options=options)


def make_linearize_logicalgap_factory(
    dem_path: Path,
    base_decoder: str,
    decoder_options: Mapping[str, Any],
    metric_options: Mapping[str, Any],
) -> DecoderFactory:
    return _LinearizeLogicalGapFactory(
        dem_path=dem_path,
        base_decoder=base_decoder,
        decoder_options=dict(decoder_options),
        metric_options=dict(metric_options),
    )
