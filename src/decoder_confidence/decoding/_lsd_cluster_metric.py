from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import stim

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._decoder_adapter import (
    _clip_priors,
    _dem_to_matrices_with_edge_priors,
)
from decoder_confidence.execution.models import DecoderBase, DecoderFactory


def _compute_cluster_llr(
    ind_cluster_stats: dict,
    w_e: np.ndarray,
    alpha: float,
) -> float:
    """
    Definition 2 (Lee, English, Bartlett 2026): Cluster LLR alpha-norm fraction.

    Q_LLR^(alpha) = [sum_i (sum_{e in C_i} w_e)^alpha]^(1/alpha) / sum_{e in E} w_e

    where w_e = log((1 - p_e) / p_e) is the prior LLR and {C_i} are the final
    active clusters from LSD (those not absorbed by another cluster).

    Returns 0.0 when there are no clusters (zero syndrome / no errors),
    which corresponds to maximum decoding confidence.
    """
    finite_mask = np.isfinite(w_e)
    total_w = float(np.sum(w_e[finite_mask]))
    if total_w == 0.0 or not ind_cluster_stats:
        return 0.0

    # Final clusters: active and not absorbed into another cluster
    active_clusters = [
        cs for cs in ind_cluster_stats.values()
        if cs.get("active", False) and cs.get("absorbed_by_cluster", -1) == -1
    ]
    if not active_clusters:
        return 0.0

    cluster_sums = []
    for cs in active_clusters:
        bits = cs.get("final_bits", [])
        if not bits:
            continue
        csum = float(np.sum(w_e[list(bits)]))
        if np.isfinite(csum):
            cluster_sums.append(csum)

    if not cluster_sums:
        return 0.0

    arr = np.array(cluster_sums, dtype=np.float64)
    if np.isinf(alpha):
        # alpha -> inf: dominated by the largest cluster
        return float(np.max(arr) / total_w)
    return float((np.sum(arr ** alpha) ** (1.0 / alpha)) / total_w)


@dataclass
class ClusterLlrDecoder(DecoderBase):
    """
    Decoder that computes the Cluster LLR alpha-norm fraction (Definition 2) as
    a confidence metric.

    Uses BP-LSD with always_run_lsd=True so that cluster statistics are
    collected on every shot regardless of BP convergence.
    """

    _decoder: Any        # ldpc.bplsd_decoder.BpLsdDecoder
    _observables: Any    # sparse matrix (num_observables, num_errors)
    _w_e: np.ndarray     # prior LLRs per error mechanism, shape (num_errors,)
    _alpha: float        # norm order for Q_LLR

    def decode(self, syndromes: np.ndarray) -> DecodingResult:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        if syndromes.ndim != 2:
            raise ValueError("syndromes must be a 1D or 2D array")

        num_shots = syndromes.shape[0]
        num_obs = int(self._observables.shape[0])

        predictions = np.zeros((num_shots, num_obs), dtype=np.bool_)
        q_llr_vals = np.zeros(num_shots, dtype=np.float64)

        for shot in range(num_shots):
            recovery = self._decoder.decode(syndromes[shot])
            stats = self._decoder.statistics
            q_llr_vals[shot] = _compute_cluster_llr(
                stats["individual_cluster_stats"], self._w_e, self._alpha
            )
            if num_obs > 0:
                predictions[shot] = (self._observables @ recovery.astype(int)) % 2

        return DecodingResult(
            predictions=predictions,
            metrics={"cluster_llr": q_llr_vals},
        )


@dataclass(frozen=True)
class _ClusterLlrFactory:
    dem_path: Path
    decoder_options: Mapping[str, Any]
    alpha: float

    def __call__(
        self, dem: stim.DetectorErrorModel | None = None
    ) -> ClusterLlrDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

        matrices = _dem_to_matrices_with_edge_priors(
            dem, allow_undecomposed_hyperedges=True
        )
        priors = _clip_priors(matrices.priors)

        # Prior LLR: w_e = log((1 - p_e) / p_e)
        # p_e = 0 yields inf (zero-probability mechanism) → treated as max confidence
        w_e = np.where(priors > 0, np.log((1.0 - priors) / priors), np.inf)

        try:
            from ldpc.bplsd_decoder import BpLsdDecoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ldpc is required for cluster_llr metric") from exc

        options = dict(self.decoder_options)
        # always_run_lsd ensures cluster stats are available even when BP converges
        options["always_run_lsd"] = True
        options.setdefault("input_vector_type", "syndrome")

        bplsd = BpLsdDecoder(
            matrices.check_matrix, error_channel=list(priors), **options
        )
        bplsd.set_do_stats(True)

        return ClusterLlrDecoder(
            _decoder=bplsd,
            _observables=matrices.observables_matrix,
            _w_e=w_e,
            _alpha=self.alpha,
        )


def make_cluster_llr_factory(
    dem_path: Path,
    decoder_options: Mapping[str, Any],
    metric_options: Mapping[str, Any],
) -> DecoderFactory:
    alpha_raw = metric_options.get("alpha", 2.0)
    alpha = float(alpha_raw) if str(alpha_raw).lower() != "inf" else np.inf
    if alpha <= 0.0:
        raise ValueError(f"metric_options.alpha must be > 0, got {alpha_raw}")
    return _ClusterLlrFactory(
        dem_path=dem_path,
        decoder_options=dict(decoder_options),
        alpha=alpha,
    )
