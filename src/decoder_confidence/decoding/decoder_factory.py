from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml

from decoder_confidence.decoding._argument_reweighting import (
    make_argument_reweighting_factory,
)
from decoder_confidence.decoding._forced_gap import make_forced_gap_ml_factory
from decoder_confidence.decoding._forcing_degradation_test import (
    FORCING_DEGRADATION_METRIC,
    make_forcing_degradation_test_factory,
)
from decoder_confidence.decoding._linearize_logicalgap import (
    make_linearize_logicalgap_factory,
)
from decoder_confidence.decoding._lsd_cluster_metric import make_cluster_llr_factory
from decoder_confidence.decoding._reweighted_linearized_gap import (
    make_reweighted_linearized_gap_factory,
)
from decoder_confidence.decoding._relay_bp import make_relay_bp_factory
from decoder_confidence.execution.models import DecoderFactory


@dataclass(frozen=True)
class DecoderConfigInfo:
    decoder_name: str
    metric_name: str
    decoder_options: Mapping[str, Any]
    metric_options: Mapping[str, Any]
    config_path: Path
    raw_config: Mapping[str, Any]



def load_decoder_factory(
    config_path: Path,
    dem_path: Path,
) -> Tuple[DecoderFactory, DecoderConfigInfo]:
    if not config_path.exists():
        raise FileNotFoundError(f"decoder_config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, Mapping):
        raise ValueError("decoder_config must be a YAML mapping")

    decoder = config.get("decoder")
    metric = config.get("metric")
    decoder_options = config.get("decoder_options") or {}
    metric_options = config.get("metric_options") or {}

    if not isinstance(decoder_options, Mapping):
        raise ValueError("decoder_options must be a mapping")
    if not isinstance(metric_options, Mapping):
        raise ValueError("metric_options must be a mapping")
    if not decoder:
        raise ValueError("decoder is required in decoder_config")
    if not metric:
        raise ValueError("metric is required in decoder_config")

    decoder_name = str(decoder).strip().upper()
    metric_name = str(metric).strip().lower()

    if metric_name == FORCING_DEGRADATION_METRIC:
        factory = make_forcing_degradation_test_factory(
            dem_path,
            decoder_name,
            decoder_options,
            metric_options,
        )
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    if metric_name in {"ar-pec", "ar_lec", "ar-lec", "ar_pec"}:
        factory = make_argument_reweighting_factory(
            dem_path,
            decoder_name,
            decoder_options,
            metric_name,
            metric_options,
        )
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    if metric_name == "forced_gap_ml":
        factory = make_forced_gap_ml_factory(
            dem_path,
            decoder_name,
            decoder_options,
            metric_options,
        )
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    if metric_name == "reweighted_linearized_gap":
        factory = make_reweighted_linearized_gap_factory(
            dem_path,
            decoder_name,
            decoder_options,
            metric_options,
        )
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    if metric_name == "linearize_logicalgap":
        factory = make_linearize_logicalgap_factory(
            dem_path,
            decoder_name,
            decoder_options,
            metric_options,
        )
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    if metric_name == "cluster_llr" and decoder_name in {"VIBE-LSD", "VIBELSD"}:
        from decoder_confidence.decoding._vibelsd import make_vibelsd_cluster_llr_factory

        factory = make_vibelsd_cluster_llr_factory(dem_path, decoder_options, metric_options)
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    if metric_name == "cluster_llr":
        if decoder_name not in {"BP-LSD", "BPLSD"}:
            raise ValueError(
                f"cluster_llr metric requires decoder: BP-LSD or VIBE-LSD, got {decoder_name}"
            )
        factory = make_cluster_llr_factory(dem_path, decoder_options, metric_options)
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    if decoder_name == "ILP":
        solver = str(decoder_options.get("solver") or "").strip().lower()
        if not solver:
            raise ValueError(
                "decoder_options.solver is required for ILP decoder. "
                "Set solver: gurobi or solver: cplex"
            )
        if solver == "gurobi":
            from decoder_confidence.decoding._ilp_logicalgap import (
                make_ilp_decoder_factory,
            )

            factory = make_ilp_decoder_factory(dem_path, decoder_options, metric_options)
        elif solver == "cplex":
            from decoder_confidence.decoding._ilp_cplex_logicalgap import (
                make_cplex_decoder_factory,
            )

            factory = make_cplex_decoder_factory(dem_path, decoder_options, metric_options)
        else:
            raise ValueError(
                f"Unknown solver '{solver}' for ILP decoder. "
                "Supported values: gurobi, cplex"
            )
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    if decoder_name in {"VIBE-LSD", "VIBELSD"}:
        from decoder_confidence.decoding._vibelsd import make_vibelsd_factory

        factory = make_vibelsd_factory(dem_path, decoder_options)
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    if decoder_name == "RELAY-BP":
        factory = make_relay_bp_factory(dem_path, decoder_options, metric_name, metric_options)
        info = DecoderConfigInfo(
            decoder_name=decoder_name,
            metric_name=metric_name,
            decoder_options=dict(decoder_options),
            metric_options=dict(metric_options),
            config_path=config_path,
            raw_config=dict(config),
        )
        return factory, info

    raise ValueError(f"Unsupported decoder/metric combination: decoder={decoder}, metric={metric}")
