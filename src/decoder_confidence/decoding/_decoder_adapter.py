from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import stim


@dataclass(frozen=True)
class DemMatricesWithEdges:
    check_matrix: Any
    observables_matrix: Any
    edge_check_matrix: Any
    edge_observables_matrix: Any
    priors: np.ndarray
    edge_priors: np.ndarray


def _clip_priors(priors: Any, *, eps: float = 1e-14) -> np.ndarray:
    values = np.asarray(priors, dtype=float)
    return np.clip(values, eps, 1.0 - eps)


def _weights_from_priors(priors: np.ndarray) -> np.ndarray:
    clipped = _clip_priors(priors)
    return np.log((1.0 - clipped) / clipped)


def _combine_prob(existing: float, prob: float) -> float:
    return existing * (1.0 - prob) + prob * (1.0 - existing)


def _iter_set_xor(set_list: list[list[int]]) -> frozenset[int]:
    out: set[int] = set()
    for items in set_list:
        current = set(items)
        out = (out - current) | (current - out)
    return frozenset(out)


def _dict_to_csc_matrix(elements_dict: dict[int, frozenset[int]], shape: tuple[int, int]):
    try:
        from scipy.sparse import csc_matrix
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scipy is required for decoder matrix construction") from exc

    nnz = sum(len(v) for v in elements_dict.values())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)

    idx = 0
    for col, rows in elements_dict.items():
        for row in rows:
            row_ind[idx] = row
            col_ind[idx] = col
            idx += 1

    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


def _dem_to_matrices_with_edge_priors(
    dem: stim.DetectorErrorModel, *, allow_undecomposed_hyperedges: bool
) -> DemMatricesWithEdges:
    hyperedge_ids: dict[frozenset[int], int] = {}
    edge_ids: dict[frozenset[int], int] = {}
    hyperedge_obs_map: dict[int, frozenset[int]] = {}
    edge_obs_map: dict[int, frozenset[int]] = {}
    priors_dict: dict[int, float] = {}
    edge_priors_dict: dict[int, float] = {}

    def handle_error(
        prob: float, detectors: list[list[int]], observables: list[list[int]]
    ) -> None:
        hyperedge_dets = _iter_set_xor(detectors)
        hyperedge_obs = _iter_set_xor(observables)

        if hyperedge_dets not in hyperedge_ids:
            hyperedge_ids[hyperedge_dets] = len(hyperedge_ids)
            priors_dict[hyperedge_ids[hyperedge_dets]] = 0.0
        hid = hyperedge_ids[hyperedge_dets]

        hyperedge_obs_map[hid] = hyperedge_obs
        priors_dict[hid] = _combine_prob(priors_dict[hid], prob)

        for i in range(len(detectors)):
            edge_dets = frozenset(detectors[i])
            edge_obs = frozenset(observables[i])

            if len(edge_dets) > 2:
                if not allow_undecomposed_hyperedges:
                    raise ValueError(
                        "A hyperedge error mechanism was not decomposed into edges. "
                        "Set decompose_errors=True when building the DEM."
                    )
                continue

            if edge_dets not in edge_ids:
                edge_ids[edge_dets] = len(edge_ids)
                edge_priors_dict[edge_ids[edge_dets]] = 0.0
            eid = edge_ids[edge_dets]
            edge_obs_map[eid] = edge_obs
            edge_priors_dict[eid] = _combine_prob(edge_priors_dict[eid], prob)

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: list[list[int]] = [[]]
            frames: list[list[int]] = [[]]
            prob = float(instruction.args_copy()[0])
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    dets[-1].append(int(target.val))
                elif target.is_logical_observable_id():
                    frames[-1].append(int(target.val))
                elif target.is_separator():
                    dets.append([])
                    frames.append([])
                else:
                    raise NotImplementedError("Unsupported DEM target type")
            handle_error(prob, dets, frames)
        elif instruction.type in {"detector", "logical_observable"}:
            continue
        else:
            raise NotImplementedError(f"Unsupported DEM instruction: {instruction.type}")

    check_matrix = _dict_to_csc_matrix(
        {v: k for k, v in hyperedge_ids.items()},
        shape=(dem.num_detectors, len(hyperedge_ids)),
    )
    observables_matrix = _dict_to_csc_matrix(
        hyperedge_obs_map,
        shape=(dem.num_observables, len(hyperedge_ids)),
    )

    edge_check_matrix = _dict_to_csc_matrix(
        {v: k for k, v in edge_ids.items()},
        shape=(dem.num_detectors, len(edge_ids)),
    )
    edge_observables_matrix = _dict_to_csc_matrix(
        edge_obs_map,
        shape=(dem.num_observables, len(edge_ids)),
    )

    priors = np.zeros(len(hyperedge_ids), dtype=float)
    for idx, prob in priors_dict.items():
        priors[idx] = prob

    edge_priors = np.zeros(len(edge_ids), dtype=float)
    for idx, prob in edge_priors_dict.items():
        edge_priors[idx] = prob

    return DemMatricesWithEdges(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        edge_check_matrix=edge_check_matrix,
        edge_observables_matrix=edge_observables_matrix,
        priors=priors,
        edge_priors=edge_priors,
    )


class DecoderAdapter(ABC):
    @property
    @abstractmethod
    def priors(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def check_matrix(self) -> Any:
        ...

    @property
    @abstractmethod
    def observables_matrix(self) -> Any:
        ...

    @property
    @abstractmethod
    def num_errors(self) -> int:
        ...

    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def set_priors(self, priors: np.ndarray) -> None:
        ...

    @abstractmethod
    def set_check_matrix(self, check_matrix: Any) -> None:
        ...

    @abstractmethod
    def set_observables(self, observables_matrix: Any) -> None:
        ...


def _dispose_solver_model(decoder: Any) -> None:
    """Best-effort release of the native solver handle held by an ILPDecoder.

    ``ILPDecoder._base_model_result.model`` is a ``cplex.Cplex`` (has
    ``.end()``) or ``gurobipy.Model`` (has ``.dispose()``); Python GC does not
    promptly release the underlying native solver state.
    """
    base = getattr(decoder, "_base_model_result", None)
    model = getattr(base, "model", None) if base is not None else None
    if model is None:
        return
    if hasattr(model, "end"):
        try:
            model.end()
        except Exception:
            pass
    elif hasattr(model, "dispose"):
        try:
            model.dispose()
        except Exception:
            pass


@dataclass
class IlpDecoderAdapter(DecoderAdapter):
    _decoder: Any
    _check_matrix: Any
    _observables_matrix: Any
    _priors: np.ndarray
    _config: Any
    _deps: Any

    @property
    def priors(self) -> np.ndarray:
        return self._priors

    @property
    def check_matrix(self) -> Any:
        return self._check_matrix

    @property
    def observables_matrix(self) -> Any:
        return self._observables_matrix

    @property
    def num_errors(self) -> int:
        return int(self._check_matrix.shape[1])

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = np.asarray(syndrome, dtype=int)
        weights = _weights_from_priors(self._priors)
        result = self._decoder.decode_result(syndrome, weight_vector=weights)
        return np.asarray(result.error_vector, dtype=np.bool_)

    def set_priors(self, priors: np.ndarray) -> None:
        self._priors = _clip_priors(priors)

    def set_check_matrix(self, check_matrix: Any) -> None:
        self._check_matrix = check_matrix
        self._rebuild()

    def set_observables(self, observables_matrix: Any) -> None:
        self._observables_matrix = observables_matrix
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            from ilp_decoder import ILPDecoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ilp_decoder is required for ILP adapter") from exc

        old_decoder = self._decoder
        self._decoder = ILPDecoder(
            parity_check_matrix=self._check_matrix,
            observables=self._observables_matrix,
            prior=self._priors,
            config=self._config,
            deps=self._deps,
        )
        _dispose_solver_model(old_decoder)


@dataclass
class BpLsdDecoderAdapter(DecoderAdapter):
    _decoder: Any
    _check_matrix: Any
    _observables_matrix: Any
    _priors: np.ndarray
    _decoder_options: Mapping[str, Any]

    @property
    def priors(self) -> np.ndarray:
        return self._priors

    @property
    def check_matrix(self) -> Any:
        return self._check_matrix

    @property
    def observables_matrix(self) -> Any:
        return self._observables_matrix

    @property
    def num_errors(self) -> int:
        return int(self._check_matrix.shape[1])

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = np.asarray(syndrome, dtype=int)
        corr = self._decoder.decode(syndrome)
        return np.asarray(corr, dtype=np.bool_)

    def set_priors(self, priors: np.ndarray) -> None:
        self._priors = _clip_priors(priors)
        if hasattr(self._decoder, "update_channel_probs"):
            self._decoder.update_channel_probs(self._priors)
        else:
            self._decoder.error_channel = self._priors

    def set_check_matrix(self, check_matrix: Any) -> None:
        self._check_matrix = check_matrix
        self._rebuild()

    def set_observables(self, observables_matrix: Any) -> None:
        self._observables_matrix = observables_matrix

    def _rebuild(self) -> None:
        self._decoder = _build_bplsd_decoder(
            self._check_matrix, self._priors, self._decoder_options
        )


@dataclass
class MwpmDecoderAdapter(DecoderAdapter):
    _matching: Any
    _check_matrix: Any
    _observables_matrix: Any
    _priors: np.ndarray
    _decoder_options: Mapping[str, Any]

    @property
    def priors(self) -> np.ndarray:
        return self._priors

    @property
    def check_matrix(self) -> Any:
        return self._check_matrix

    @property
    def observables_matrix(self) -> Any:
        return self._observables_matrix

    @property
    def num_errors(self) -> int:
        return int(self._check_matrix.shape[1])

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        corr = self._matching.decode(syndrome)
        return np.asarray(corr, dtype=np.bool_)

    def set_priors(self, priors: np.ndarray) -> None:
        self._priors = _clip_priors(priors)
        self._matching = _build_matching(
            self._check_matrix, self._priors, self._decoder_options
        )

    def set_check_matrix(self, check_matrix: Any) -> None:
        self._check_matrix = check_matrix
        self._matching = _build_matching(
            self._check_matrix, self._priors, self._decoder_options
        )

    def set_observables(self, observables_matrix: Any) -> None:
        self._observables_matrix = observables_matrix


def _build_decoder_config(options: Mapping[str, Any]):
    try:
        from ilp_decoder import DecoderConfig
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ilp_decoder is required for ILP adapter") from exc

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
        raise RuntimeError("gurobipy is required for ILP adapter") from exc

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if log_to_console else 0)
    env.start()
    return env


def _build_ilp_adapter(
    dem: stim.DetectorErrorModel,
    decoder_options: Mapping[str, Any],
) -> IlpDecoderAdapter:
    try:
        from ilp_decoder import ILPDecoder
        from ilp_decoder.core import DecoderDependencies
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ilp_decoder is required for ILP adapter") from exc

    matrices = _dem_to_matrices_with_edge_priors(
        dem, allow_undecomposed_hyperedges=True
    )
    priors = _clip_priors(matrices.priors)
    config = _build_decoder_config(decoder_options)
    env = _build_gurobi_env(config.log_to_console)
    deps = DecoderDependencies(env=env)
    decoder = ILPDecoder(
        parity_check_matrix=matrices.check_matrix,
        observables=matrices.observables_matrix,
        prior=priors,
        config=config,
        deps=deps,
    )
    return IlpDecoderAdapter(
        _decoder=decoder,
        _check_matrix=matrices.check_matrix,
        _observables_matrix=matrices.observables_matrix,
        _priors=priors,
        _config=config,
        _deps=deps,
    )


def _build_bplsd_decoder(
    check_matrix: Any,
    priors: np.ndarray,
    decoder_options: Mapping[str, Any],
):
    try:
        from ldpc.bplsd_decoder import BpLsdDecoder
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ldpc is required for BP-LSD adapter") from exc

    options = dict(decoder_options)
    options.setdefault("input_vector_type", "syndrome")

    return BpLsdDecoder(check_matrix, error_channel=list(priors), **options)


def _build_bplsd_adapter(
    dem: stim.DetectorErrorModel,
    decoder_options: Mapping[str, Any],
) -> BpLsdDecoderAdapter:
    matrices = _dem_to_matrices_with_edge_priors(
        dem, allow_undecomposed_hyperedges=True
    )
    priors = _clip_priors(matrices.priors)
    decoder = _build_bplsd_decoder(matrices.check_matrix, priors, decoder_options)
    return BpLsdDecoderAdapter(
        _decoder=decoder,
        _check_matrix=matrices.check_matrix,
        _observables_matrix=matrices.observables_matrix,
        _priors=priors,
        _decoder_options=dict(decoder_options),
    )


def _build_matching(
    check_matrix: Any,
    priors: np.ndarray,
    decoder_options: Mapping[str, Any],
):
    try:
        import pymatching
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pymatching is required for MWPM adapter") from exc

    options = dict(decoder_options)
    for key in ("weights", "spacelike_weights", "error_probabilities", "faults_matrix"):
        if key in options:
            raise ValueError(f"MWPM adapter manages '{key}' internally")

    weights = _weights_from_priors(priors)
    return pymatching.Matching.from_check_matrix(check_matrix, weights=weights, **options)


def _build_mwpm_adapter(
    dem: stim.DetectorErrorModel, decoder_options: Mapping[str, Any]
) -> MwpmDecoderAdapter:
    # allow_undecomposed_hyperedges=False: raise if the DEM contains 3+-detector
    # hyperedges that MWPM cannot represent as graph edges.  For DEM-filtered cases
    # (xyz_decoding=False surface/color code) the DEM has only ≤2-detector errors,
    # so check_matrix == edge_check_matrix and this always succeeds.
    matrices = _dem_to_matrices_with_edge_priors(
        dem, allow_undecomposed_hyperedges=False
    )
    priors = _clip_priors(matrices.priors)
    matching = _build_matching(matrices.check_matrix, priors, decoder_options)
    return MwpmDecoderAdapter(
        _matching=matching,
        _check_matrix=matrices.check_matrix,
        _observables_matrix=matrices.observables_matrix,
        _priors=priors,
        _decoder_options=dict(decoder_options),
    )


@dataclass
class RelayBpDecoderAdapter(DecoderAdapter):
    _check_matrix: Any
    _observables_matrix: Any
    _priors: np.ndarray
    _decoder_options: Mapping[str, Any]
    _decoder: Any  # relay_bp.bp.RelayDecoderF64

    @property
    def priors(self) -> np.ndarray:
        return self._priors

    @property
    def check_matrix(self) -> Any:
        return self._check_matrix

    @property
    def observables_matrix(self) -> Any:
        return self._observables_matrix

    @property
    def num_errors(self) -> int:
        return int(self._check_matrix.shape[1])

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        result = self.decode_detailed_single(syndrome)
        return np.asarray(result.decoding, dtype=np.bool_)

    def decode_detailed_single(self, syndrome: np.ndarray) -> Any:
        syndrome_u8 = np.asarray(syndrome, dtype=np.uint8)
        return self._decoder.decode_detailed(syndrome_u8)

    def set_priors(self, priors: np.ndarray) -> None:
        self._priors = _clip_priors(priors)
        self._rebuild()

    def set_check_matrix(self, check_matrix: Any) -> None:
        self._check_matrix = check_matrix
        self._rebuild()

    def set_observables(self, observables_matrix: Any) -> None:
        self._observables_matrix = observables_matrix

    def _rebuild(self) -> None:
        self._decoder = _build_relay_decoder(
            self._check_matrix, self._priors, self._decoder_options
        )


def _build_relay_decoder(
    check_matrix: Any,
    priors: np.ndarray,
    decoder_options: Mapping[str, Any],
) -> Any:
    try:
        from relay_bp.bp import RelayDecoderF64
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("relay_bp is required for relay-bp adapter") from exc

    options = dict(decoder_options)
    gamma_dist = options.get("gamma_dist_interval")
    if gamma_dist is not None:
        options["gamma_dist_interval"] = tuple(gamma_dist)

    return RelayDecoderF64(
        check_matrix=check_matrix,
        error_priors=np.asarray(priors, dtype=np.float64),
        **options,
    )


def _build_relay_bp_adapter(
    dem: stim.DetectorErrorModel,
    decoder_options: Mapping[str, Any],
) -> RelayBpDecoderAdapter:
    matrices = _dem_to_matrices_with_edge_priors(
        dem, allow_undecomposed_hyperedges=True
    )
    priors = _clip_priors(matrices.priors)
    decoder = _build_relay_decoder(matrices.check_matrix, priors, decoder_options)
    return RelayBpDecoderAdapter(
        _check_matrix=matrices.check_matrix,
        _observables_matrix=matrices.observables_matrix,
        _priors=priors,
        _decoder_options=dict(decoder_options),
        _decoder=decoder,
    )


def _normalize_decoder_name(name: str) -> str:
    return name.strip().upper().replace("_", "-")


def build_decoder_adapter(
    decoder_name: str,
    dem: stim.DetectorErrorModel,
    decoder_options: Mapping[str, Any],
) -> DecoderAdapter:
    normalized = _normalize_decoder_name(decoder_name)
    if normalized == "ILP":
        return _build_ilp_adapter(dem, decoder_options)
    if normalized in {"BP-LSD", "BPLSD"}:
        return _build_bplsd_adapter(dem, decoder_options)
    if normalized in {"MWPM", "PYMATCHING"}:
        return _build_mwpm_adapter(dem, decoder_options)

    if normalized in {"VIBE-LSD", "VIBELSD"}:
        from decoder_confidence.decoding._vibelsd import build_vibelsd_adapter
        return build_vibelsd_adapter(dem, decoder_options)

    if normalized == "RELAY-BP":
        return _build_relay_bp_adapter(dem, decoder_options)

    raise ValueError(f"Unsupported base decoder for AR: {decoder_name}")
