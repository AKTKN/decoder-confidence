from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import stim

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._decoder_adapter import (
    RelayBpDecoderAdapter,
    _build_relay_bp_adapter,
    _clip_priors,
    _weights_from_priors,
)
from decoder_confidence.decoding._linearize_logicalgap import (
    _get_obs_row,
    _append_row,
    _logical_from_correction,
)
from decoder_confidence.decoding._argument_reweighting import (
    _reweight_ratio,
    _reweight_gap,
)
from decoder_confidence.execution.models import DecoderBase, DecoderFactory

FALLBACK_GAP: float = -32768.0
FALLBACK_FORCED_GAP: float = 0.0

_SUPPORTED_METRICS = frozenset(
    {"linearized_logicalgap", "forced_gap_ml", "reweighted_linearized_gap", "argument_reweighting"}
)


@dataclass(frozen=True)
class _RelayBpAROptions:
    test_type: str   # "Ratio" or "Gap"
    b: float
    num_decoding_rounds: int
    criterion: str   # "PEC" or "LEC"

    def validate(self) -> None:
        if self.num_decoding_rounds < 2:
            raise ValueError("num_decoding_rounds must be >= 2")
        if self.criterion not in {"PEC", "LEC"}:
            raise ValueError("criterion must be 'pec' or 'lec'")
        if self.test_type not in {"Ratio", "Gap"}:
            raise ValueError("test_type must be 'Ratio' or 'Gap'")
        if self.test_type == "Ratio" and self.b <= 1.0:
            raise ValueError("b must be > 1 for Ratio test")
        if self.test_type == "Gap" and self.b <= 0.0:
            raise ValueError("b must be > 0 for Gap test")


@dataclass(frozen=True)
class _RelayBpRLGOptions:
    b: float

    def validate(self) -> None:
        if self.b <= 1.0:
            raise ValueError("b must be > 1 for ratio reweighting")


def _parse_relay_bp_metric_options(
    metric: str, metric_options: Mapping[str, Any]
) -> Any:
    if metric == "argument_reweighting":
        test_type = metric_options.get("test_type")
        b = metric_options.get("b")
        num_rounds = metric_options.get("num_decoding_rounds")
        criterion = metric_options.get("criterion")
        if test_type is None:
            raise ValueError("metric_options.test_type is required for argument_reweighting")
        if b is None:
            raise ValueError("metric_options.b is required for argument_reweighting")
        if num_rounds is None:
            raise ValueError("metric_options.num_decoding_rounds is required for argument_reweighting")
        if criterion is None:
            raise ValueError("metric_options.criterion is required for argument_reweighting")
        opts = _RelayBpAROptions(
            test_type=str(test_type).strip().title(),
            b=float(b),
            num_decoding_rounds=int(num_rounds),
            criterion=str(criterion).strip().upper(),
        )
        opts.validate()
        return opts
    if metric == "reweighted_linearized_gap":
        b = metric_options.get("b")
        if b is None:
            raise ValueError("metric_options.b is required for reweighted_linearized_gap")
        opts = _RelayBpRLGOptions(b=float(b))
        opts.validate()
        return opts
    return None


@dataclass
class RelayBpMetricDecoder(DecoderBase):
    adapter: RelayBpDecoderAdapter
    metric: str
    parsed_options: Any  # _RelayBpAROptions | _RelayBpRLGOptions | None

    def __post_init__(self) -> None:
        if self.metric not in _SUPPORTED_METRICS:
            raise ValueError(f"Unsupported relay-bp metric: {self.metric}")
        self._base_priors: np.ndarray = _clip_priors(self.adapter.priors)
        self._base_check_matrix: Any = self.adapter.check_matrix
        self._observables: Any = self.adapter.observables_matrix
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
        metric_values: list[Any] = []
        obs_flip_idx: list[list[int]] = []

        for shot in range(num_shots):
            syndrome = syndromes[shot]
            if self.metric == "linearized_logicalgap":
                pred, mv, ofi = self._shot_linearized_logicalgap(syndrome, num_obs)
            elif self.metric == "forced_gap_ml":
                pred, mv, ofi = self._shot_forced_gap_ml(syndrome, num_obs)
            elif self.metric == "reweighted_linearized_gap":
                pred, mv, ofi = self._shot_reweighted_linearized_gap(syndrome, num_obs)
            else:  # argument_reweighting
                pred, mv, ofi = self._shot_argument_reweighting(syndrome)
            predictions[shot] = pred
            metric_values.append(mv)
            obs_flip_idx.append(ofi)

        return DecodingResult(
            predictions=predictions,
            metrics={self.metric: np.asarray(metric_values)},
            obs_flip_idx=obs_flip_idx,
        )

    def _stage1(self, syndrome: np.ndarray):
        self.adapter.set_priors(self._base_priors.copy())
        self.adapter.set_check_matrix(self._base_check_matrix)
        result = self.adapter.decode_detailed_single(syndrome)
        c1 = np.asarray(result.decoding, dtype=np.bool_)
        l1 = _logical_from_correction(self._observables, c1)
        w1 = float(self._weights @ c1.astype(int))
        return result.success, c1, l1, w1

    def _stage2_constrained(self, syndrome: np.ndarray, l1: np.ndarray, num_obs: int):
        """Run constrained decodes for each observable. Returns list of (w, l, converged)."""
        results = []
        for i in range(num_obs):
            obs_row_i = _get_obs_row(self._observables, i)
            aug_check = _append_row(self._base_check_matrix, obs_row_i)
            aug_syndrome = np.append(syndrome, 1 - int(l1[i]))
            self.adapter.set_check_matrix(aug_check)
            self.adapter.set_priors(self._base_priors.copy())
            r = self.adapter.decode_detailed_single(aug_syndrome)
            c2 = np.asarray(r.decoding, dtype=np.bool_)
            l2 = _logical_from_correction(self._observables, c2)
            w2 = float(self._weights @ c2.astype(int))
            results.append((w2, l2, r.success))
        self.adapter.set_check_matrix(self._base_check_matrix)
        self.adapter.set_priors(self._base_priors.copy())
        return results

    def _shot_linearized_logicalgap(
        self, syndrome: np.ndarray, num_obs: int
    ) -> tuple[np.ndarray, float, list[int]]:
        converged1, c1, l1, w1 = self._stage1(syndrome)
        pred = l1.astype(np.bool_)

        if not converged1 or num_obs == 0:
            return pred, FALLBACK_GAP, []

        stage2 = self._stage2_constrained(syndrome, l1, num_obs)
        converged_w2 = [w for w, _, ok in stage2 if ok]

        if not converged_w2:
            return pred, FALLBACK_GAP, []

        best_w2 = min(converged_w2)
        # obs_flip_idx: find best converged solution's logical diff vs l1
        best_flip: list[int] = []
        for w, l2, ok in stage2:
            if ok and w == best_w2:
                diff = np.asarray(l1, dtype=int) ^ np.asarray(l2, dtype=int)
                best_flip = list(np.where(diff)[0])
                break

        return pred, float(best_w2 - w1), best_flip

    def _shot_forced_gap_ml(
        self, syndrome: np.ndarray, num_obs: int
    ) -> tuple[np.ndarray, float, list[int]]:
        converged1, c1, l1, w1 = self._stage1(syndrome)

        if not converged1 or num_obs == 0:
            return l1.astype(np.bool_), FALLBACK_FORCED_GAP, []

        stage2 = self._stage2_constrained(syndrome, l1, num_obs)
        converged_s2 = [(w, l) for w, l, ok in stage2 if ok]

        if not converged_s2:
            return l1.astype(np.bool_), FALLBACK_FORCED_GAP, []

        all_solutions: list[tuple[float, np.ndarray]] = [(w1, l1)] + converged_s2
        all_solutions.sort(key=lambda x: x[0])

        ml_weight, ml_logical = all_solutions[0]
        pred = ml_logical.astype(np.bool_)

        best_diff_weight = np.inf
        for w, l in all_solutions[1:]:
            if not np.array_equal(l, ml_logical):
                best_diff_weight = w
                break

        gap = (best_diff_weight - ml_weight) if np.isfinite(best_diff_weight) else np.nan
        diff = np.asarray(l1, dtype=int) ^ np.asarray(ml_logical, dtype=int)
        ofi = list(np.where(diff)[0])
        return pred, float(gap) if not np.isnan(gap) else float("nan"), ofi

    def _shot_reweighted_linearized_gap(
        self, syndrome: np.ndarray, num_obs: int
    ) -> tuple[np.ndarray, float, list[int]]:
        opts: _RelayBpRLGOptions = self.parsed_options
        converged1, c1, l1, w1 = self._stage1(syndrome)
        pred = l1.astype(np.bool_)

        if not converged1 or num_obs == 0:
            return pred, FALLBACK_GAP, []

        # Stage 2a: constrained decodes
        stage2a = self._stage2_constrained(syndrome, l1, num_obs)
        converged_2a = [(w, l) for w, l, ok in stage2a if ok]

        if not converged_2a:
            # No 2a converged → worst confidence regardless of 2b
            return pred, FALLBACK_GAP, []

        best_w_constrained = min(w for w, _ in converged_2a)
        # Track best_flip for obs_flip_idx
        best_flip: list[int] = []
        for w, l2 in converged_2a:
            if w == best_w_constrained:
                diff = np.asarray(l1, dtype=int) ^ np.asarray(l2, dtype=int)
                best_flip = list(np.where(diff)[0])
                break

        # Stage 2b: reweighted unconstrained decode
        reweighted_priors = _clip_priors(
            _reweight_ratio(self._base_priors.copy(), opts.b, c1)
        )
        self.adapter.set_check_matrix(self._base_check_matrix)
        self.adapter.set_priors(reweighted_priors)
        r_b = self.adapter.decode_detailed_single(syndrome)
        self.adapter.set_priors(self._base_priors.copy())

        if r_b.success:
            c_r = np.asarray(r_b.decoding, dtype=np.bool_)
            l_r = _logical_from_correction(self._observables, c_r)
            w_r = float(self._weights @ c_r.astype(int))
            l_r_same = bool(np.array_equal(l_r, l1))

            if l_r_same and w_r < w1:
                gap = best_w_constrained - w_r
            elif l_r_same:
                gap = best_w_constrained - w1
            else:
                gap = min(best_w_constrained, w_r) - w1
        else:
            # 2b not converged: only 2a results, case a
            gap = best_w_constrained - w1

        return pred, float(gap), best_flip

    def _shot_argument_reweighting(
        self, syndrome: np.ndarray
    ) -> tuple[np.ndarray, bool, list[int]]:
        opts: _RelayBpAROptions = self.parsed_options
        num_obs = int(self._observables.shape[0])

        priors = self._base_priors.copy()
        self.adapter.set_priors(priors)
        self.adapter.set_check_matrix(self._base_check_matrix)

        r0 = self.adapter.decode_detailed_single(syndrome)
        c0 = np.asarray(r0.decoding, dtype=np.bool_)
        l0 = _logical_from_correction(self._observables, c0).astype(np.bool_)
        pred = l0

        if not r0.success:
            return pred, False, []

        if not c0.any():
            return pred, True, []

        accepted = True
        prev_c = c0
        for _ in range(1, opts.num_decoding_rounds):
            if opts.test_type == "Ratio":
                priors = _clip_priors(_reweight_ratio(priors, opts.b, prev_c))
            else:
                priors = _clip_priors(_reweight_gap(priors, opts.b, prev_c))
            self.adapter.set_priors(priors)
            ri = self.adapter.decode_detailed_single(syndrome)

            if not ri.success:
                accepted = False
                break

            ci = np.asarray(ri.decoding, dtype=np.bool_)
            if opts.criterion == "PEC":
                if not np.array_equal(ci, c0):
                    accepted = False
                    break
            else:
                li = _logical_from_correction(self._observables, ci)
                if not np.array_equal(li, l0):
                    accepted = False
                    break
            prev_c = ci

        self.adapter.set_priors(self._base_priors.copy())
        return pred, bool(accepted), []


def _validate_relay_bp_metric(metric: str, metric_options: Mapping[str, Any]) -> None:
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"relay-bp supports metrics: {sorted(_SUPPORTED_METRICS)}, got '{metric}'"
        )
    _parse_relay_bp_metric_options(metric, metric_options)


@dataclass(frozen=True)
class _RelayBpFactory:
    dem_path: Path
    decoder_options: Mapping[str, Any]
    metric: str
    metric_options: Mapping[str, Any]

    def __call__(
        self, dem: stim.DetectorErrorModel | None = None
    ) -> RelayBpMetricDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))
        adapter = _build_relay_bp_adapter(dem, self.decoder_options)
        parsed_options = _parse_relay_bp_metric_options(self.metric, self.metric_options)
        return RelayBpMetricDecoder(
            adapter=adapter,
            metric=self.metric,
            parsed_options=parsed_options,
        )


def make_relay_bp_factory(
    dem_path: Path,
    decoder_options: Mapping[str, Any],
    metric: str,
    metric_options: Mapping[str, Any],
) -> DecoderFactory:
    _validate_relay_bp_metric(metric, dict(metric_options))
    return _RelayBpFactory(
        dem_path=dem_path,
        decoder_options=dict(decoder_options),
        metric=metric,
        metric_options=dict(metric_options),
    )
