from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import stim

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._decoder_adapter import (
    DecoderAdapter,
    RelayBpDecoderAdapter,
    build_decoder_adapter,
    _clip_priors,
    _weights_from_priors,
)
from decoder_confidence.decoding._linearize_logicalgap import (
    _get_obs_row,
    _append_row,
    _logical_from_correction,
)
from decoder_confidence.decoding._argument_reweighting import _reweight_ratio
from decoder_confidence.execution.models import DecoderBase, DecoderFactory

_FORCED_UNCONVERGED_CHOICES = frozenset({"positive", "negative"})


@dataclass(frozen=True)
class ReweightedLinearizedGapOptions:
    b: float
    forced_unconverged_confidence_value: str | None = None  # "positive" or "negative"

    def validate(self) -> None:
        if self.b <= 1.0:
            raise ValueError("b must be > 1 for ratio reweighting")


def _parse_rlg_options(metric_options: Mapping[str, Any]) -> ReweightedLinearizedGapOptions:
    b = metric_options.get("b")
    if b is None:
        raise ValueError("metric_options.b is required for reweighted_linearized_gap")

    forced_unconverged_raw = metric_options.get("forced_unconverged_confidence_value")
    forced_unconverged: str | None = None
    if forced_unconverged_raw is not None:
        forced_unconverged = str(forced_unconverged_raw).strip().lower()
        if forced_unconverged not in _FORCED_UNCONVERGED_CHOICES:
            raise ValueError(
                "forced_unconverged_confidence_value must be 'positive' or 'negative'"
            )

    options = ReweightedLinearizedGapOptions(
        b=float(b), forced_unconverged_confidence_value=forced_unconverged
    )
    options.validate()
    return options


@dataclass
class ReweightedLinearizedGapDecoder(DecoderBase):
    """Two-stage decoder combining ratio reweighting and linearized logical gap.

    Stage 1: decode normally → E_base (c1, w1, l1).
    Stage 2a: for each observable i, decode with augmented parity constraint forcing a
              flip, yielding constrained solutions. E_second = min-weight among these.
    Stage 2b: decode with ratio-reweighted priors (p_e^b for bits set in c1), no
              constraint → E_r (c_r, w_r, l_r). Weights are always under base priors.

    Metric cases (w_r and w1 compared under base priors):
      a. l_r == l1 and w1 <= w_r  →  metric = w_second - w1
      b. l_r == l1 and w_r < w1   →  metric = w_second - w_r
      c. l_r != l1                 →  metric = min(w_second, w_r) - w1
    """

    adapter: DecoderAdapter
    options: ReweightedLinearizedGapOptions

    def __post_init__(self) -> None:
        self.options.validate()
        self._base_priors: np.ndarray = _clip_priors(self.adapter.priors)
        self._base_check_matrix: Any = self.adapter.check_matrix
        self._observables: Any = self.adapter.observables_matrix
        self._num_errors: int = self.adapter.num_errors
        self._weights: np.ndarray = _weights_from_priors(self._base_priors)
        self._relay_bp_adapter = isinstance(self.adapter, RelayBpDecoderAdapter)
        if self._relay_bp_adapter and self.options.forced_unconverged_confidence_value is None:
            raise ValueError(
                "metric_options.forced_unconverged_confidence_value is required when "
                "using relay-bp with the reweighted_linearized_gap metric"
            )

    def _decode_shot(self, syndrome: np.ndarray) -> tuple[np.ndarray, bool]:
        if self._relay_bp_adapter:
            assert isinstance(self.adapter, RelayBpDecoderAdapter)
            result = self.adapter.decode_detailed_single(syndrome)
            return np.asarray(result.decoding, dtype=np.bool_), bool(result.success)
        return np.asarray(self.adapter.decode(syndrome), dtype=np.bool_), True

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
        force_logical_error = (
            np.zeros((num_shots,), dtype=np.bool_) if self._relay_bp_adapter else None
        )

        for shot in range(num_shots):
            syndrome = syndromes[shot]

            # --- Stage 1: decode with original check matrix and priors ---
            self.adapter.set_priors(self._base_priors.copy())
            self.adapter.set_check_matrix(self._base_check_matrix)
            c1, converged1 = self._decode_shot(syndrome)

            l1 = _logical_from_correction(self._observables, c1)
            w1 = float(self._weights @ c1.astype(int))
            predictions[shot] = l1.astype(np.bool_)

            if self._relay_bp_adapter and not converged1:
                gap[shot] = -np.inf
                obs_flip_idx.append([])
                assert force_logical_error is not None
                force_logical_error[shot] = True
                continue

            forced_value = (
                np.inf if self.options.forced_unconverged_confidence_value == "positive" else -np.inf
            )

            if num_obs == 0:
                if self._relay_bp_adapter:
                    gap[shot] = forced_value
                obs_flip_idx.append([])
                continue

            # --- Stage 2a: constrained instances, one per observable ---
            best_w_constrained = np.inf
            best_flip: list[int] = []
            have_2a = False
            for i in range(num_obs):
                obs_row_i = _get_obs_row(self._observables, i)
                aug_check = _append_row(self._base_check_matrix, obs_row_i)
                target_bit = 1 - int(l1[i])
                aug_syndrome = np.append(syndrome, target_bit)

                self.adapter.set_check_matrix(aug_check)
                self.adapter.set_priors(self._base_priors.copy())
                c2_i, converged2_i = self._decode_shot(aug_syndrome)

                w2_i = float(self._weights @ c2_i.astype(int))
                eligible = (not self._relay_bp_adapter) or converged2_i
                if eligible:
                    have_2a = True
                    if w2_i < best_w_constrained:
                        best_w_constrained = w2_i
                        l2_i = _logical_from_correction(self._observables, c2_i)
                        diff = np.asarray(l1, dtype=int) ^ np.asarray(l2_i, dtype=int)
                        best_flip = list(np.where(diff)[0])

            # --- Stage 2b: reweighted unconstrained instance ---
            reweighted_priors = _clip_priors(
                _reweight_ratio(self._base_priors.copy(), self.options.b, c1)
            )
            self.adapter.set_check_matrix(self._base_check_matrix)
            self.adapter.set_priors(reweighted_priors)
            c_r, converged_r = self._decode_shot(syndrome)

            l_r = _logical_from_correction(self._observables, c_r)
            w_r = float(self._weights @ c_r.astype(int))  # weight under base priors

            # Restore state for next shot
            self.adapter.set_check_matrix(self._base_check_matrix)
            self.adapter.set_priors(self._base_priors.copy())

            b_converged = (not self._relay_bp_adapter) or converged_r
            l_r_same = b_converged and bool(np.array_equal(l_r, l1))
            have_2b_diff = b_converged and not l_r_same

            if self._relay_bp_adapter and not have_2a and not have_2b_diff:
                gap[shot] = forced_value
                obs_flip_idx.append([])
                continue

            # --- Metric computation ---
            if b_converged:
                if l_r_same and w_r < w1:
                    # Case b: E_r is a better correction in the same logical class
                    effective_base_w = w_r
                    w_second = best_w_constrained
                elif l_r_same:
                    # Case a: E_base is the better correction in the same logical class
                    effective_base_w = w1
                    w_second = best_w_constrained
                else:
                    # Case c: E_r is in a different logical class; include in "other
                    # class" pool
                    effective_base_w = w1
                    w_second = min(best_w_constrained, w_r)
                    if not have_2a:
                        diff = np.asarray(l1, dtype=int) ^ np.asarray(l_r, dtype=int)
                        best_flip = list(np.where(diff)[0])
            else:
                # Stage 2b did not converge: only Stage 2a results are usable.
                effective_base_w = w1
                w_second = best_w_constrained

            gap[shot] = w_second - effective_base_w
            obs_flip_idx.append(best_flip)

        metrics: dict[str, Any] = {"reweighted_linearized_gap": gap}
        if force_logical_error is not None:
            metrics["__is_logical_error"] = force_logical_error

        return DecodingResult(
            predictions=predictions,
            metrics=metrics,
            obs_flip_idx=obs_flip_idx,
        )


@dataclass(frozen=True)
class _ReweightedLinearizedGapFactory:
    dem_path: Path
    base_decoder: str
    decoder_options: Mapping[str, Any]
    metric_options: Mapping[str, Any]

    def __call__(
        self, dem: stim.DetectorErrorModel | None = None
    ) -> ReweightedLinearizedGapDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

        adapter = build_decoder_adapter(self.base_decoder, dem, self.decoder_options)
        options = _parse_rlg_options(self.metric_options)

        return ReweightedLinearizedGapDecoder(adapter=adapter, options=options)


def make_reweighted_linearized_gap_factory(
    dem_path: Path,
    base_decoder: str,
    decoder_options: Mapping[str, Any],
    metric_options: Mapping[str, Any],
) -> DecoderFactory:
    return _ReweightedLinearizedGapFactory(
        dem_path=dem_path,
        base_decoder=base_decoder,
        decoder_options=dict(decoder_options),
        metric_options=dict(metric_options),
    )
