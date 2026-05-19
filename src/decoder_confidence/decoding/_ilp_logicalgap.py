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
from decoder_confidence.execution.models import DecoderBase, DecoderFactory


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
        else:
            predictions = np.zeros((0, 0), dtype=bool)
            logical_gap = np.zeros((0,), dtype=float)

        return DecodingResult(
            predictions=predictions,
            metrics={"logical_gap": logical_gap},
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


def _build_gurobi_env(log_to_console: bool):
    try:
        import gurobipy as gp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("gurobipy is required for ILP decoder") from exc

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if log_to_console else 0)
    env.start()
    return env


@dataclass(frozen=True)
class _ILPDecoderFactory:
    dem_path: Path
    decoder_options: Mapping[str, Any]
    use_edge_matrices: bool = False

    def __call__(self, dem: stim.DetectorErrorModel | None = None) -> _ILPLogicalGapDecoder:
        config = _build_decoder_config(self.decoder_options)
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

        matrices = _dem_to_matrices_with_edge_priors(dem, allow_undecomposed_hyperedges=True)
        if self.use_edge_matrices:
            parity_check_matrix = matrices.edge_check_matrix
            observables = matrices.edge_observables_matrix
            priors = _clip_priors(matrices.edge_priors)
        else:
            parity_check_matrix = matrices.check_matrix
            observables = matrices.observables_matrix
            priors = _clip_priors(matrices.priors)

        env = _build_gurobi_env(config.log_to_console)
        deps = DecoderDependencies(env=env)
        decoder = ILPDecoder(
            parity_check_matrix=parity_check_matrix,
            observables=observables,
            prior=priors,
            config=config,
            deps=deps,
        )
        return _ILPLogicalGapDecoder(decoder=decoder)


def make_ilp_decoder_factory(
    dem_path: Path,
    decoder_options: Mapping[str, Any],
    use_edge_matrices: bool = False,
) -> DecoderFactory:
    # Use a top-level callable to keep the factory picklable for multiprocessing spawn.
    return _ILPDecoderFactory(
        dem_path=dem_path,
        decoder_options=dict(decoder_options),
        use_edge_matrices=use_edge_matrices,
    )