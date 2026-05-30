from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import stim

from ilp_decoder import DecoderConfig, ILPDecoder
from ilp_decoder.core import DecoderDependencies

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._decoder_adapter import (
    _clip_priors,
    _dem_to_matrices_with_edge_priors,
)
from decoder_confidence.execution.models import DecoderBase, SharedEnvDecoderFactory


@dataclass
class _ILPLogicalGapDecoder(DecoderBase):
    decoder: ILPDecoder

    def decode(self, syndromes: np.ndarray) -> DecodingResult:
        syndromes = np.asarray(syndromes, dtype=int)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        if syndromes.ndim != 2:
            raise ValueError("syndromes must be 1D or 2D array")

        results = self.decoder.decode_batch_result(syndromes, get_logicalgap=True, logical_gap_flip_last_detector=False)
        if results:
            predictions = np.stack(
                [np.asarray(result.predicted_observables, dtype=bool) for result in results],
                axis=0,
            )
            logical_gap = np.array(
                [
                    float(result.metadata.get("logical_gap"))
                    if result.metadata.get("logical_gap") is not None
                    else np.nan
                    for result in results
                ],
                dtype=float,
            )
            obs_flip_idx = [
                list(result.metadata.get("obs_flip_idx") or [])
                for result in results
            ]
        else:
            predictions = np.zeros((0, 0), dtype=bool)
            logical_gap = np.zeros((0,), dtype=float)
            obs_flip_idx = []

        return DecodingResult(
            predictions=predictions,
            metrics={"logical_gap": logical_gap},
            obs_flip_idx=obs_flip_idx,
        )


def _build_decoder_config(options: Mapping[str, Any]) -> DecoderConfig:
    time_limit_s = options.get("time_limit_s")
    mip_gap = options.get("mip_gap")
    threads = options.get("threads")
    if threads is None:
        threads = 1
    log_to_console = bool(options.get("log_to_console", False))
    enable_lazy_constraints = bool(options.get("enable_lazy_constraints", False))
    random_seed = options.get("random_seed")
    solver_options = options.get("solver_options") or {}

    if not isinstance(solver_options, Mapping):
        raise ValueError("solver_options must be a mapping")

    return DecoderConfig(
        time_limit_s=time_limit_s,
        mip_gap=mip_gap,
        threads=threads,
        log_to_console=log_to_console,
        enable_lazy_constraints=enable_lazy_constraints,
        random_seed=random_seed,
        solver_options=solver_options,
    )


def _build_ilp_env(log_to_console: bool):
    """Create and start a Gurobi env configured for long-running simulations.

    WLSTokenDuration=30 and WLSTokenRefresh=0.5 ensure the token is
    automatically renewed every 30 min for the lifetime of the process.
    """
    try:
        import gurobipy as gp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("gurobipy is required for ILP decoder") from exc

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if log_to_console else 0)
    env.setParam("WLSTokenDuration", 20)
    env.setParam("WLSTokenRefresh", 0.5)
    env.start()
    return env


def _build_ilp_decoder_with_env(
    env: Any, dem: stim.DetectorErrorModel, config: DecoderConfig
) -> _ILPLogicalGapDecoder:
    matrices = _dem_to_matrices_with_edge_priors(dem, allow_undecomposed_hyperedges=True)
    priors = _clip_priors(matrices.priors)
    deps = DecoderDependencies(env=env)
    decoder = ILPDecoder(
        parity_check_matrix=matrices.check_matrix,
        observables=matrices.observables_matrix,
        prior=priors,
        config=config,
        deps=deps,
    )
    return _ILPLogicalGapDecoder(decoder=decoder)


@dataclass(frozen=True)
class _ILPDecoderFactory(SharedEnvDecoderFactory):
    """Factory for ILP-based logical-gap decoder.

    Extends SharedEnvDecoderFactory so the execution manager creates a single
    Gurobi env (= 1 WLS session) and passes it to each worker thread.
    """

    dem_path: Path
    decoder_options: Mapping[str, Any]

    def build_env(self) -> Any:
        config = _build_decoder_config(self.decoder_options)
        return _build_ilp_env(config.log_to_console)

    def build_decoder(self, env: Any, dem: stim.DetectorErrorModel) -> _ILPLogicalGapDecoder:
        config = _build_decoder_config(self.decoder_options)
        return _build_ilp_decoder_with_env(env, dem, config)

    def __call__(self, dem: stim.DetectorErrorModel | None = None) -> _ILPLogicalGapDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))
        return self.build_decoder(self.build_env(), dem)


def make_ilp_decoder_factory(
    dem_path: Path,
    decoder_options: Mapping[str, Any],
) -> _ILPDecoderFactory:
    # Use a top-level callable to keep the factory picklable for multiprocessing spawn.
    return _ILPDecoderFactory(
        dem_path=dem_path,
        decoder_options=dict(decoder_options),
    )
