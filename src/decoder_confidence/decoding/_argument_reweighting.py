from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import stim

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._decoder_adapter import DecoderAdapter, build_decoder_adapter
from decoder_confidence.execution.models import DecoderBase, DecoderFactory


@dataclass(frozen=True)
class ArgumentReweightingOptions:
	criterion: str
	test_type: str
	b: float
	num_decoding_rounds: int

	def validate(self) -> None:
		if self.num_decoding_rounds < 2:
			raise ValueError("num_decoding_rounds must be >= 2")
		if self.criterion not in {"PEC", "LEC"}:
			raise ValueError("criterion must be 'PEC' or 'LEC'")
		if self.test_type not in {"Ratio", "Gap"}:
			raise ValueError("test_type must be 'Ratio' or 'Gap'")
		if self.test_type == "Ratio" and self.b <= 1.0:
			raise ValueError("b must be > 1 for Ratio test")
		if self.test_type == "Gap" and self.b <= 0.0:
			raise ValueError("b must be > 0 for Gap test")


def _clip_priors(priors: np.ndarray, eps: float = 1e-14) -> np.ndarray:
	return np.clip(priors, eps, 1.0 - eps)


def _parse_criterion(metric_name: str) -> str:
	normalized = metric_name.strip().lower().replace("_", "-")
	if normalized == "ar-pec":
		return "PEC"
	if normalized == "ar-lec":
		return "LEC"
	raise ValueError(f"Unsupported AR metric: {metric_name}")


def _parse_ar_options(
	metric_name: str, metric_options: Mapping[str, Any]
) -> ArgumentReweightingOptions:
	test_type = metric_options.get("test_type")
	b = metric_options.get("b")
	num_rounds = metric_options.get("num_decoding_rounds")

	if test_type is None:
		raise ValueError("metric_options.test_type is required for AR")
	if b is None:
		raise ValueError("metric_options.b is required for AR")
	if num_rounds is None:
		raise ValueError("metric_options.num_decoding_rounds is required for AR")

	options = ArgumentReweightingOptions(
		criterion=_parse_criterion(metric_name),
		test_type=str(test_type).strip().title(),
		b=float(b),
		num_decoding_rounds=int(num_rounds),
	)
	options.validate()
	return options


def _reweight_ratio(priors: np.ndarray, b: float, correction: np.ndarray) -> np.ndarray:
	updated = priors.copy()
	mask = correction.astype(bool)
	updated[mask] = np.power(updated[mask], b)
	return updated


def _reweight_gap(priors: np.ndarray, b: float, correction: np.ndarray) -> np.ndarray:
	updated = priors.copy()
	mask = correction.astype(bool)
	if not mask.any():
		return updated
	log_p_c = np.log(updated[mask]).sum()
	if log_p_c == 0.0:
		raise ValueError("log p(c) is zero; gap test undefined")
	scale = np.exp(-(np.log(updated[mask]) / log_p_c) * b)
	updated[mask] = updated[mask] * scale
	return updated


def _apply_reweight(
	priors: np.ndarray, options: ArgumentReweightingOptions, correction: np.ndarray
) -> np.ndarray:
	if options.test_type == "Ratio":
		return _reweight_ratio(priors, options.b, correction)
	return _reweight_gap(priors, options.b, correction)


def _logical_from_c(observables: Any, correction: np.ndarray) -> np.ndarray:
	num_obs = int(observables.shape[0])
	if num_obs == 0:
		return np.zeros((0,), dtype=np.bool_)

	try:
		from scipy import sparse
	except ImportError:  # pragma: no cover - optional dependency
		sparse = None

	vec = np.asarray(correction, dtype=int)
	if sparse is not None and sparse.issparse(observables):
		return (observables @ vec) % 2
	obs_dense = np.asarray(observables, dtype=int)
	return (obs_dense @ vec) % 2


@dataclass
class ArgumentReweightingDecoder(DecoderBase):
	adapter: DecoderAdapter
	metric_name: str
	options: ArgumentReweightingOptions

	def __post_init__(self) -> None:
		self.options.validate()
		self._base_priors = _clip_priors(self.adapter.priors)
		self._observables = self.adapter.observables_matrix
		self._num_errors = self.adapter.num_errors

	def decode(self, syndromes: np.ndarray) -> DecodingResult:
		syndromes = np.asarray(syndromes, dtype=int)
		if syndromes.ndim == 1:
			syndromes = syndromes.reshape(1, -1)
		if syndromes.ndim != 2:
			raise ValueError("syndromes must be 1D or 2D array")

		num_shots = syndromes.shape[0]
		num_obs = int(self._observables.shape[0])

		predictions = np.zeros((num_shots, num_obs), dtype=np.bool_)
		accept = np.zeros((num_shots,), dtype=np.bool_)

		for shot in range(num_shots):
			syndrome = syndromes[shot]
			priors = self._base_priors.copy()
			self.adapter.set_priors(priors)

			c0 = np.asarray(self.adapter.decode(syndrome), dtype=np.bool_)
			if c0.shape[0] != self._num_errors:
				raise ValueError("decoder correction length does not match priors")

			base_logical = _logical_from_c(self._observables, c0).astype(np.bool_)
			predictions[shot] = base_logical

			if not c0.any():
				accept[shot] = True
				continue

			accepted = True
			prev_c = c0
			for _ in range(1, self.options.num_decoding_rounds):
				priors = _clip_priors(_apply_reweight(priors, self.options, prev_c))
				self.adapter.set_priors(priors)
				ci = np.asarray(self.adapter.decode(syndrome), dtype=np.bool_)

				if self.options.criterion == "PEC":
					if not np.array_equal(ci, c0):
						accepted = False
						break
				else:
					logical_i = _logical_from_c(self._observables, ci)
					if not np.array_equal(logical_i, base_logical):
						accepted = False
						break
				prev_c = ci

			accept[shot] = True if accepted else False

		return DecodingResult(
			predictions=predictions,
			metrics={self.metric_name: accept},
		)


@dataclass(frozen=True)
class _ArgumentReweightingFactory:
	dem_path: Path
	base_decoder: str
	decoder_options: Mapping[str, Any]
	metric_name: str
	metric_options: Mapping[str, Any]

	def __call__(
		self, dem: stim.DetectorErrorModel | None = None
	) -> ArgumentReweightingDecoder:
		if dem is None:
			dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

		adapter = build_decoder_adapter(self.base_decoder, dem, self.decoder_options)
		options = _parse_ar_options(self.metric_name, self.metric_options)

		return ArgumentReweightingDecoder(
			adapter=adapter,
			metric_name=self.metric_name,
			options=options,
		)


def make_argument_reweighting_factory(
	dem_path: Path,
	base_decoder: str,
	decoder_options: Mapping[str, Any],
	metric_name: str,
	metric_options: Mapping[str, Any],
) -> DecoderFactory:
	return _ArgumentReweightingFactory(
		dem_path=dem_path,
		base_decoder=base_decoder,
		decoder_options=dict(decoder_options),
		metric_name=metric_name,
		metric_options=dict(metric_options),
	)
