from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import stim

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._decoder_adapter import (
    DecoderAdapter,
    _clip_priors,
    _dem_to_matrices_with_edge_priors,
    _weights_from_priors,
)
from decoder_confidence.execution.models import DecoderBase, DecoderFactory


# ---------------------------------------------------------------------------
# §1  Options & parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VibeLsdOptions:
    ensemble_size: int = 32
    correction_limit: int = 5
    max_iter: int = 25
    bp_method: str = "minimum_sum"
    ms_scaling_factor: float = 0.625
    lsd_order: int = 0
    prior_perturbation: float = 0.0  # δ: perturbation width per instance
    seed: int = 0
    always_run_lsd: bool = False  # internal — not parsed from YAML


def _parse_vibelsd_options(decoder_options: Mapping[str, Any]) -> VibeLsdOptions:
    opts = dict(decoder_options)
    opts.pop("always_run_lsd", None)  # never read from YAML
    return VibeLsdOptions(
        ensemble_size=int(opts.get("ensemble_size", 32)),
        correction_limit=int(opts.get("correction_limit", 5)),
        max_iter=int(opts.get("max_iter", 25)),
        bp_method=str(opts.get("bp_method", "minimum_sum")),
        ms_scaling_factor=float(opts.get("ms_scaling_factor", 0.625)),
        lsd_order=int(opts.get("lsd_order", 0)),
        prior_perturbation=float(opts.get("prior_perturbation", 0.0)),
        seed=int(opts.get("seed", 0)),
        always_run_lsd=False,
    )


def _options_with_always_run_lsd(options: VibeLsdOptions) -> VibeLsdOptions:
    return VibeLsdOptions(
        ensemble_size=options.ensemble_size,
        correction_limit=options.correction_limit,
        max_iter=options.max_iter,
        bp_method=options.bp_method,
        ms_scaling_factor=options.ms_scaling_factor,
        lsd_order=options.lsd_order,
        prior_perturbation=options.prior_perturbation,
        seed=options.seed,
        always_run_lsd=True,
    )


# ---------------------------------------------------------------------------
# §2  Offline setup (pure functions)
# ---------------------------------------------------------------------------


def _make_permutations(
    n_errors: int, ensemble_size: int, rng: np.random.Generator
) -> list[list[int]]:
    """Return L distinct random permutations of [0, n_errors)."""
    return [rng.permutation(n_errors).tolist() for _ in range(ensemble_size)]


def _build_bp_ensemble(
    check_matrix: Any,
    priors: np.ndarray,
    permutations: list[list[int]],
    options: VibeLsdOptions,
) -> list[Any]:
    """Build one BpDecoder per permutation, each with a fixed serial schedule."""
    try:
        from ldpc import BpDecoder
    except ImportError as exc:
        raise RuntimeError("ldpc is required for VibeLSD") from exc

    return [
        BpDecoder(
            check_matrix,
            error_channel=list(priors),
            max_iter=options.max_iter,
            bp_method=options.bp_method,
            ms_scaling_factor=options.ms_scaling_factor,
            schedule="serial",
            serial_schedule_order=perm,
            input_vector_type="syndrome",
        )
        for perm in permutations
    ]


def _build_lsd_fallback(
    check_matrix: Any,
    options: VibeLsdOptions,
    *,
    do_stats: bool = False,
) -> Any:
    """Build the LSD fallback decoder.

    When do_stats=False (normal fallback path): use LsdDecoder, which takes
    (syndrome, bit_weights) directly and avoids an extra BP step.

    When do_stats=True (cluster_llr path): use BpLsdDecoder with max_iter=1
    and always_run_lsd=True.  LsdDecoder does not expose cluster statistics,
    so BpLsdDecoder is the only ldpc option that does.  One BP iteration
    slightly modifies the averaged LLRs before they reach LSD, but this is
    the only practical way to obtain cluster statistics in Python.
    """
    try:
        from ldpc.lsd_decoder import LsdDecoder
        from ldpc.bplsd_decoder import BpLsdDecoder
    except ImportError as exc:
        raise RuntimeError("ldpc is required for VibeLSD") from exc

    if do_stats:
        n = check_matrix.shape[1]
        placeholder = [0.1] * n  # updated at decode time via error_channel
        lsd = BpLsdDecoder(
            check_matrix,
            error_channel=placeholder,
            max_iter=1,
            always_run_lsd=True,
            lsd_order=options.lsd_order,
            input_vector_type="syndrome",
        )
        lsd.set_do_stats(True)
        return lsd

    return LsdDecoder(check_matrix, lsd_order=options.lsd_order)


# ---------------------------------------------------------------------------
# §3  Per-syndrome decoding steps (pure functions)
# ---------------------------------------------------------------------------


def _perturb_priors(
    priors: np.ndarray,
    delta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample p̃_e ~ Uniform[(1-δ)p_e, (1+δ)p_e] independently per mechanism."""
    if delta == 0.0:
        return priors
    lo = np.maximum(0.0, (1.0 - delta) * priors)
    hi = np.minimum(1.0, (1.0 + delta) * priors)
    return rng.uniform(lo, hi)


def _run_bp_instance(
    bp: Any,
    syndrome: np.ndarray,
    channel: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Set channel probs, run BP, return (converged, correction, log_prob_ratios)."""
    bp.error_channel = list(channel)
    correction = bp.decode(syndrome)
    converged = bool(bp.converge)
    llrs = np.asarray(bp.log_prob_ratios, dtype=np.float64)
    return converged, np.asarray(correction, dtype=np.bool_), llrs


def _collect_candidates(
    bp_instances: list[Any],
    syndrome: np.ndarray,
    base_priors: np.ndarray,
    correction_limit: int,
    delta: float,
    rng: np.random.Generator,
    *,
    run_all: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run the BP ensemble on a single syndrome.

    Returns (candidates, all_llrs):
      candidates  list of converged corrections (up to correction_limit unless run_all)
      all_llrs    LLR vector from every BP instance that was run

    run_all=True skips early termination so that all L LLR vectors are
    collected — necessary for the cluster_llr path where LSD always runs.
    """
    candidates: list[np.ndarray] = []
    all_llrs: list[np.ndarray] = []

    for bp in bp_instances:
        channel = _perturb_priors(base_priors, delta, rng)
        converged, correction, llrs = _run_bp_instance(bp, syndrome, channel)
        all_llrs.append(llrs)
        if converged:
            candidates.append(correction)
            if not run_all and len(candidates) >= correction_limit:
                break

    return candidates, all_llrs


def _select_best_correction(
    candidates: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Return the candidate with minimum Σ w_e · c_e (maximum likelihood)."""
    best_cost = np.inf
    best = candidates[0]
    for c in candidates:
        cost = float(np.dot(weights, c.astype(np.float64)))
        if cost < best_cost:
            best_cost = cost
            best = c
    return best


def _average_normalized_llrs(llr_list: list[np.ndarray]) -> np.ndarray:
    """Compute (1/L) Σ_i  LLR^i / ‖LLR^i‖_2."""
    result = np.zeros(len(llr_list[0]), dtype=np.float64)
    for llrs in llr_list:
        norm = float(np.sqrt(np.dot(llrs, llrs)))
        if norm > 0.0:
            result += llrs / norm
    return result / len(llr_list)


def _decode_with_lsd(
    lsd: Any,
    syndrome: np.ndarray,
    soft_weights: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Run LSD with averaged soft information; return (correction, stats_dict).

    Handles two backends:
      LsdDecoder    — takes (syndrome, bit_weights) directly; no statistics.
      BpLsdDecoder  — set error_channel from converted LLRs, then decode;
                      cluster statistics available via .statistics.
    """
    if hasattr(lsd, "error_channel"):  # BpLsdDecoder
        probs = _clip_priors(1.0 / (1.0 + np.exp(soft_weights)))
        lsd.error_channel = list(probs)
        correction = lsd.decode(syndrome)
        stats: dict = lsd.statistics or {}
    else:  # LsdDecoder
        correction = lsd.decode(syndrome, soft_weights)
        stats = {}

    return np.asarray(correction, dtype=np.bool_), stats


# ---------------------------------------------------------------------------
# §4  VibeLsdDecoderAdapter
# ---------------------------------------------------------------------------


class VibeLsdDecoderAdapter(DecoderAdapter):
    """DecoderAdapter wrapping the VibeLSD ensemble-BP + LSD pipeline.

    Drop-in replacement for BpLsdDecoderAdapter inside
    ArgumentReweightingDecoder and LinearizeLogicalGapDecoder.

    Prior perturbation (δ) is applied entirely inside decode().  Callers
    such as ArgumentReweightingDecoder interact only with unperturbed priors
    via set_priors(), which stores AR-modified priors as the perturbation
    centre without triggering a rebuild.
    """

    def __init__(
        self,
        check_matrix: Any,
        observables_matrix: Any,
        base_priors: np.ndarray,
        options: VibeLsdOptions,
        permutations: list[list[int]],
        rng: np.random.Generator,
    ) -> None:
        self._check_matrix = check_matrix
        self._observables_matrix = observables_matrix
        self._base_priors = _clip_priors(base_priors)
        self._options = options
        self._permutations = permutations
        self._rng = rng
        self._weights = _weights_from_priors(self._base_priors)
        self._last_lsd_stats: dict | None = None

        self._bp_instances = _build_bp_ensemble(
            check_matrix, self._base_priors, permutations, options
        )
        self._lsd = _build_lsd_fallback(
            check_matrix, options, do_stats=options.always_run_lsd
        )

    # ------------------------------------------------------------------
    # DecoderAdapter properties
    # ------------------------------------------------------------------

    @property
    def priors(self) -> np.ndarray:
        return self._base_priors

    @property
    def check_matrix(self) -> Any:
        return self._check_matrix

    @property
    def observables_matrix(self) -> Any:
        return self._observables_matrix

    @property
    def num_errors(self) -> int:
        return int(self._check_matrix.shape[1])

    @property
    def last_lsd_stats(self) -> dict | None:
        """Cluster statistics from the most recent LSD call (cluster_llr mode only)."""
        return self._last_lsd_stats

    # ------------------------------------------------------------------
    # DecoderAdapter mutators
    # ------------------------------------------------------------------

    def set_priors(self, priors: np.ndarray) -> None:
        """Update base priors (e.g. from argument reweighting).

        Stores the modified priors as the new perturbation centre.
        Does not rebuild the ensemble — perturbation is applied fresh
        inside decode() on every call.
        """
        self._base_priors = _clip_priors(priors)
        self._weights = _weights_from_priors(self._base_priors)

    def set_check_matrix(self, check_matrix: Any) -> None:
        # linearize_logicalgap appends one row per observable, so the column
        # count (N = num error mechanisms) stays constant and the stored
        # permutations remain valid.  Rebuilding L BP decoders + LSD is
        # O(L) more expensive than a single BP-LSD rebuild.
        self._check_matrix = check_matrix
        self._rebuild_ensemble()

    def set_observables(self, observables_matrix: Any) -> None:
        self._observables_matrix = observables_matrix

    # ------------------------------------------------------------------
    # Core decode
    # ------------------------------------------------------------------

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome_u8 = np.asarray(syndrome, dtype=np.uint8)
        run_all = self._options.always_run_lsd

        candidates, all_llrs = _collect_candidates(
            self._bp_instances,
            syndrome_u8,
            self._base_priors,
            self._options.correction_limit,
            self._options.prior_perturbation,
            self._rng,
            run_all=run_all,
        )

        if self._options.always_run_lsd:
            avg_llrs = _average_normalized_llrs(all_llrs)
            lsd_correction, stats = _decode_with_lsd(self._lsd, syndrome_u8, avg_llrs)
            self._last_lsd_stats = stats
            return _select_best_correction(candidates, self._weights) if candidates else lsd_correction

        if candidates:
            return _select_best_correction(candidates, self._weights)

        avg_llrs = _average_normalized_llrs(all_llrs)
        lsd_correction, _ = _decode_with_lsd(self._lsd, syndrome_u8, avg_llrs)
        return lsd_correction

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_ensemble(self) -> None:
        self._bp_instances = _build_bp_ensemble(
            self._check_matrix, self._base_priors, self._permutations, self._options
        )
        self._lsd = _build_lsd_fallback(
            self._check_matrix, self._options, do_stats=self._options.always_run_lsd
        )


# ---------------------------------------------------------------------------
# §5  Standalone decoder, cluster_llr decoder, and factories
# ---------------------------------------------------------------------------


@dataclass
class VibeLsdStandaloneDecoder(DecoderBase):
    _adapter: VibeLsdDecoderAdapter

    def decode(self, syndromes: np.ndarray) -> DecodingResult:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        if syndromes.ndim != 2:
            raise ValueError("syndromes must be 1D or 2D")

        num_shots = syndromes.shape[0]
        num_obs = int(self._adapter.observables_matrix.shape[0])
        predictions = np.zeros((num_shots, num_obs), dtype=np.bool_)

        for shot in range(num_shots):
            correction = self._adapter.decode(syndromes[shot])
            if num_obs > 0:
                predictions[shot] = (
                    self._adapter.observables_matrix @ correction.astype(int)
                ) % 2

        return DecodingResult(predictions=predictions, metrics={})


@dataclass
class VibeLsdClusterLlrDecoder(DecoderBase):
    """VibeLSD decoder that computes the Cluster LLR alpha-norm fraction.

    LSD always runs (using averaged normalized ensemble LLRs as soft weights)
    so that cluster statistics are available on every shot.  The error
    prediction is taken from the best converged BP correction when available,
    falling back to LSD otherwise — identical to VibeLSD's standard decode.
    """

    _adapter: VibeLsdDecoderAdapter  # constructed with always_run_lsd=True
    _w_e: np.ndarray                 # DEM prior LLRs, shape (num_errors,)
    _alpha: float

    def decode(self, syndromes: np.ndarray) -> DecodingResult:
        from decoder_confidence.decoding._lsd_cluster_metric import _compute_cluster_llr

        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        if syndromes.ndim != 2:
            raise ValueError("syndromes must be 1D or 2D")

        num_shots = syndromes.shape[0]
        num_obs = int(self._adapter.observables_matrix.shape[0])
        predictions = np.zeros((num_shots, num_obs), dtype=np.bool_)
        q_llr_vals = np.zeros(num_shots, dtype=np.float64)

        for shot in range(num_shots):
            correction = self._adapter.decode(syndromes[shot])
            if num_obs > 0:
                predictions[shot] = (
                    self._adapter.observables_matrix @ correction.astype(int)
                ) % 2
            stats = self._adapter.last_lsd_stats or {}
            q_llr_vals[shot] = _compute_cluster_llr(
                stats.get("individual_cluster_stats", {}),
                self._w_e,
                self._alpha,
            )

        return DecodingResult(
            predictions=predictions,
            metrics={"cluster_llr": q_llr_vals},
        )


# ---------------------------------------------------------------------------
# Factory helpers (called from decoder_factory.py and _decoder_adapter.py)
# ---------------------------------------------------------------------------


def build_vibelsd_adapter(
    dem: stim.DetectorErrorModel,
    decoder_options: Mapping[str, Any],
    *,
    always_run_lsd: bool = False,
) -> VibeLsdDecoderAdapter:
    matrices = _dem_to_matrices_with_edge_priors(
        dem, allow_undecomposed_hyperedges=True
    )
    priors = _clip_priors(matrices.priors)
    options = _parse_vibelsd_options(decoder_options)
    if always_run_lsd:
        options = _options_with_always_run_lsd(options)

    rng = np.random.default_rng(options.seed)
    n_errors = int(matrices.check_matrix.shape[1])
    permutations = _make_permutations(n_errors, options.ensemble_size, rng)

    return VibeLsdDecoderAdapter(
        check_matrix=matrices.check_matrix,
        observables_matrix=matrices.observables_matrix,
        base_priors=priors,
        options=options,
        permutations=permutations,
        rng=rng,
    )


@dataclass(frozen=True)
class _VibeLsdFactory:
    dem_path: Path
    decoder_options: Mapping[str, Any]

    def __call__(
        self, dem: stim.DetectorErrorModel | None = None
    ) -> VibeLsdStandaloneDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))
        adapter = build_vibelsd_adapter(dem, self.decoder_options)
        return VibeLsdStandaloneDecoder(_adapter=adapter)


@dataclass(frozen=True)
class _VibeLsdClusterLlrFactory:
    dem_path: Path
    decoder_options: Mapping[str, Any]
    alpha: float

    def __call__(
        self, dem: stim.DetectorErrorModel | None = None
    ) -> VibeLsdClusterLlrDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

        matrices = _dem_to_matrices_with_edge_priors(
            dem, allow_undecomposed_hyperedges=True
        )
        priors = _clip_priors(matrices.priors)
        w_e = np.where(priors > 0, np.log((1.0 - priors) / priors), np.inf)

        adapter = build_vibelsd_adapter(dem, self.decoder_options, always_run_lsd=True)
        return VibeLsdClusterLlrDecoder(_adapter=adapter, _w_e=w_e, _alpha=self.alpha)


def make_vibelsd_factory(
    dem_path: Path,
    decoder_options: Mapping[str, Any],
) -> DecoderFactory:
    return _VibeLsdFactory(dem_path=dem_path, decoder_options=dict(decoder_options))


def make_vibelsd_cluster_llr_factory(
    dem_path: Path,
    decoder_options: Mapping[str, Any],
    metric_options: Mapping[str, Any],
) -> DecoderFactory:
    alpha_raw = metric_options.get("alpha", 2.0)
    alpha = float(alpha_raw) if str(alpha_raw).lower() != "inf" else np.inf
    if alpha <= 0.0:
        raise ValueError(f"metric_options.alpha must be > 0, got {alpha_raw}")
    return _VibeLsdClusterLlrFactory(
        dem_path=dem_path,
        decoder_options=dict(decoder_options),
        alpha=alpha,
    )
