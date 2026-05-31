from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import stim

from ilp_decoder import DecoderConfig, ILPDecoder
from ilp_decoder.core import DecoderDependencies, OptimizerBackend
from ilp_decoder.cplex_backend import CplexEnv
from ilp_decoder.cplex_formulation import (
    build_base_model_cplex,
    build_logical_gap_model_cplex,
)

from decoder_confidence.decoding._ilp_logicalgap import _ILPLogicalGapDecoder
from decoder_confidence.decoding._decoder_adapter import (
    _clip_priors,
    _dem_to_matrices_with_edge_priors,
)
from decoder_confidence.execution.models import DecoderBase


def _build_cplex_decoder_config(options: Mapping[str, Any]) -> DecoderConfig:
    threads = options.get("threads")
    if threads is None:
        threads = 1
    time_limit_s = options.get("time_limit_s")
    mip_gap = options.get("mip_gap")
    log_to_console = bool(options.get("log_to_console", False))
    enable_lazy_constraints = bool(options.get("enable_lazy_constraints", False))
    random_seed = options.get("random_seed")

    return DecoderConfig(
        time_limit_s=time_limit_s,
        mip_gap=mip_gap,
        threads=threads,
        log_to_console=log_to_console,
        enable_lazy_constraints=enable_lazy_constraints,
        random_seed=random_seed,
    )


def _build_base_model_cplex_with_symmetry(
    env: CplexEnv,
    parity_check_matrix: Any,
    observables: Any,
    config: DecoderConfig,
    *,
    prior: Any = None,
    weight_vector: Any = None,
    extra_constraints: Any = None,
) -> Any:
    result = build_base_model_cplex(
        env,
        parity_check_matrix,
        observables,
        config,
        prior=prior,
        weight_vector=weight_vector,
        extra_constraints=extra_constraints,
    )
    result.model.parameters.preprocessing.symmetry.set(4)
    return result


def _build_logical_gap_model_cplex_with_symmetry(
    env: CplexEnv,
    parity_check_matrix: Any,
    observables: Any,
    config: DecoderConfig,
    *,
    logical_class: Any,
    prior: Any = None,
    weight_vector: Any = None,
    extra_constraints: Any = None,
) -> Any:
    result = build_logical_gap_model_cplex(
        env,
        parity_check_matrix,
        observables,
        config,
        logical_class=logical_class,
        prior=prior,
        weight_vector=weight_vector,
        extra_constraints=extra_constraints,
    )
    result.model.parameters.preprocessing.symmetry.set(4)
    return result


def _build_cplex_decoder(
    dem: stim.DetectorErrorModel, config: DecoderConfig
) -> _ILPLogicalGapDecoder:
    matrices = _dem_to_matrices_with_edge_priors(dem, allow_undecomposed_hyperedges=True)
    priors = _clip_priors(matrices.priors)
    env = CplexEnv()
    deps = DecoderDependencies(
        env=env,
        backend=OptimizerBackend.CPLEX,
        model_builder=_build_base_model_cplex_with_symmetry,
        gap_model_builder=_build_logical_gap_model_cplex_with_symmetry,
    )
    decoder = ILPDecoder(
        parity_check_matrix=matrices.check_matrix,
        observables=matrices.observables_matrix,
        prior=priors,
        config=config,
        deps=deps,
    )
    return _ILPLogicalGapDecoder(decoder=decoder)


@dataclass(frozen=True)
class _CPLEXDecoderFactory:
    """Factory for CPLEX-based ILP logical-gap decoder.

    Unlike the Gurobi factory, CPLEX with an academic licence has no session
    limits, so this is a plain callable factory (not SharedEnvDecoderFactory).
    The execution manager will route it to the multiprocessing pool path.
    """

    dem_path: Path
    decoder_options: Mapping[str, Any]

    def __call__(self, dem: stim.DetectorErrorModel | None = None) -> _ILPLogicalGapDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))
        config = _build_cplex_decoder_config(self.decoder_options)
        return _build_cplex_decoder(dem, config)


def make_cplex_decoder_factory(
    dem_path: Path,
    decoder_options: Mapping[str, Any],
) -> _CPLEXDecoderFactory:
    return _CPLEXDecoderFactory(
        dem_path=dem_path,
        decoder_options=dict(decoder_options),
    )
