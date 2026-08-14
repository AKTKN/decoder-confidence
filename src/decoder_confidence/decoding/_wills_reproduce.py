from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import stim

from decoder_confidence.config import DecodingResult
from decoder_confidence.decoding._constraints import (
    ConstrainedDecodeOptions,
    build_constrained_system,
)
from decoder_confidence.decoding._decoder_adapter import (
    RelayBpDecoderAdapter,
    _clip_priors,
    _weights_from_priors,
    build_decoder_adapter,
)
from decoder_confidence.decoding._linearize_logicalgap import (
    _get_obs_row,
    _logical_from_correction,
)
from decoder_confidence.execution.models import DecoderBase, DecoderFactory

# Metric name used in configs (``metric: wills_reproduce``) and in the
# output parquet/directory naming (``decoder=RELAY-BP,metric=wills_reproduce,...``).
WILLS_REPRODUCE_METRIC = "wills_reproduce"

# The paper (Wills, Yoder & Chuang, "Forced Gap Post-Selection for Quantum
# LDPC Codes and their Operations", arXiv:2605.20346) does not decimate the
# high-weight row appended to build each forced run's H^(i) -- they note
# that as future work (Outlook, "... by using decimation to break up the
# high-weight row of the H^(i) matrices that may be impeding convergence").
# random_split=False reproduces that (matches _constraints.py's naive
# single-row _append_row path).
_CONSTRAINED_OPTIONS = ConstrainedDecodeOptions()


@dataclass(frozen=True)
class WillsReproduceOptions:
    """Options for the ``wills_reproduce`` metric.

    ``forced_num_sets`` overrides ``decoder_options["num_sets"]`` for the K
    forced runs only -- the baseline run always uses
    ``decoder_options["num_sets"]`` unmodified. This mirrors Appendix A of
    the paper, where the baseline and forced runs use different ``R``
    (``num_sets``) values (e.g. 1201 vs 25 for the 72-qubit idling code)
    while sharing every other Relay-BP parameter (``gamma0``,
    ``gamma_dist_interval``, ``pre_iter``, ``set_max_iter``, ``stop_nconv``,
    ``seed``).
    """

    forced_num_sets: int
    get_detail_stat: bool = False


def _parse_wills_reproduce_options(metric_options: Mapping[str, Any]) -> WillsReproduceOptions:
    options = dict(metric_options)
    allowed = {"forced_num_sets", "get_detail_stat"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            "Unsupported wills_reproduce metric option(s): "
            + ", ".join(unknown)
            + f". Supported options: {', '.join(sorted(allowed))}"
        )

    if "forced_num_sets" not in options:
        raise ValueError(
            "metric_options.forced_num_sets is required for wills_reproduce. "
            "The paper uses a much smaller num_sets (R) for the K forced runs "
            "than for the baseline run (e.g. 25 vs 1201 for the 72-qubit "
            "idling code) -- see Appendix A."
        )
    forced_num_sets = int(options["forced_num_sets"])
    if forced_num_sets < 1:
        raise ValueError(f"metric_options.forced_num_sets must be >= 1, got {forced_num_sets}")

    return WillsReproduceOptions(
        forced_num_sets=forced_num_sets,
        get_detail_stat=bool(options.get("get_detail_stat", False)),
    )


@dataclass
class WillsReproduceDecoder(DecoderBase):
    """Reproduction of the "forced gap" post-selection strategy of Wills,
    Yoder & Chuang, "Forced Gap Post-Selection for Quantum LDPC Codes and
    their Operations" (arXiv:2605.20346), Methods section ("Forced gap
    post-selection", Fig. 1) -- kept as a module separate from
    ``forced_gap_ml`` (``_forced_gap.py``) since its Relay-BP configuration,
    decoder-count, and unconverged-case convention are all fixed by the
    paper rather than user-configurable.

    Phase 1 (baseline run): decode normally with ``baseline_adapter`` ->
    correction e0, logical class L0 = A . e0, weight w0 (a stand-in for
    -log P[e0]).

    Phase 2 (K forced runs): for each observable i, force the decoder to
    find a correction whose i'th logical bit differs from L0 -- this is the
    same constrained-system construction ``forced_gap_ml`` uses
    (``build_constrained_system``) -- but decoded with a *separate*
    ``forced_adapter`` (``num_sets=forced_num_sets`` in place of the
    baseline's ``num_sets``), matching the paper's differentiated R between
    baseline and forced runs (Appendix A).

    Phase 3: pool the baseline + up-to-K forced candidates (set Lambda in
    the paper), take each distinct logical class's best (lowest-weight)
    representative, and set
        Gap = weight(second-best distinct class) - weight(best class).
    The predicted logical class is the best (lowest-weight) one overall,
    which may differ from L0 if some forced run found a lower-weight
    solution (matching the paper's Gap definition, Eq. 4).

    Unconverged cases follow the paper's fixed convention (not
    user-configurable, unlike ``forced_gap_ml``'s
    ``forced_unconverged_confidence_value``):

    * baseline run does not converge -> Lambda is empty ("erasure"),
      Gap = 0, and the shot is forced to count as a logical error
      (``__is_logical_error``). Since Gap = 0 is the lowest possible value,
      any post-selection threshold T > 0 rejects it; at T = 0 it is counted
      as a logical error, matching "erasure events are considered logical
      errors" (no post-selection) and "erasures are always rejected when T
      is positive" in the paper.
    * baseline converges but none of the K forced runs converge -> Gap =
      +inf, and the shot is always accepted, matching "non-erasure
      instances where all forced runs were unconverged are always
      accepted, since then Gap = inf by convention".

    Known simplification vs. the paper: each Relay-BP call here returns
    only the single best-of-its-legs solution
    (``RelayBpDecoderAdapter.decode_detailed_single``), not every converged
    leg's candidate solution. The paper explicitly pools candidates from
    *all* converged legs of *every* call, including the baseline's, into
    Lambda ("we indeed use all solutions in our simulations") -- the
    Relay-BP implementation vendored in this repo only tracks/returns the
    single best solution per call (see ``decode_detailed`` in
    ``relay_too_slow/relay/crates/relay_bp/src/bp/relay.rs``), so Lambda
    here has at most K+1 entries (one per call) rather than up to
    (K+1) x stop_nconv. Using a large ``stop_nconv`` (e.g. 100, as in
    Appendix A) still makes each of those K+1 entries a much better
    (closer to true-minimum-weight) estimate than the production
    ``forced_gap_ml`` run's ``stop_nconv=1``.
    """

    baseline_adapter: RelayBpDecoderAdapter
    forced_adapter: RelayBpDecoderAdapter
    options: WillsReproduceOptions

    def __post_init__(self) -> None:
        self._base_priors: np.ndarray = _clip_priors(self.baseline_adapter.priors)
        self._base_check_matrix: Any = self.baseline_adapter.check_matrix
        self._observables: Any = self.baseline_adapter.observables_matrix
        self._weights: np.ndarray = _weights_from_priors(self._base_priors)
        self._partition_cache: dict[int, Any] = {}

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
        force_logical_error = np.zeros((num_shots,), dtype=np.bool_)

        get_detail_stat = self.options.get_detail_stat
        if get_detail_stat:
            baseline_iteration = np.full((num_shots,), np.nan, dtype=float)
            forced_iteration = np.full((num_shots,), np.nan, dtype=float)
            forced_2nd_best_iteration = np.full((num_shots,), np.nan, dtype=float)
            num_forced_converged = np.zeros((num_shots,), dtype=np.int64)

        for shot in range(num_shots):
            syndrome = syndromes[shot]

            # --- Phase 1: baseline run ---
            self.baseline_adapter.set_check_matrix_and_priors(
                self._base_check_matrix,
                self._base_priors.copy(),
            )
            result0 = self.baseline_adapter.decode_detailed_single(syndrome)
            converged0 = bool(result0.success)
            c0 = np.asarray(result0.decoding, dtype=np.bool_)
            l0 = _logical_from_correction(self._observables, c0)
            w0 = float(self._weights @ c0.astype(int))
            if get_detail_stat:
                baseline_iteration[shot] = float(result0.iterations) if converged0 else np.nan

            if not converged0:
                # Erasure (Lambda empty): Gap = 0, forced logical error.
                gap[shot] = 0.0
                predictions[shot] = l0.astype(np.bool_)
                obs_flip_idx.append([])
                force_logical_error[shot] = True
                continue

            all_solutions: list[tuple[float, np.ndarray]] = [(w0, l0)]
            forced_results: list[tuple[float, np.ndarray, float]] = []

            if num_obs > 0:
                # --- Phase 2: K forced runs ---
                for i in range(num_obs):
                    obs_row_i = _get_obs_row(self._observables, i)
                    target_bit = 1 - int(l0[i])
                    constrained = build_constrained_system(
                        self._base_check_matrix,
                        syndrome,
                        self._base_priors,
                        obs_row_i,
                        target_bit,
                        _CONSTRAINED_OPTIONS,
                        self._partition_cache,
                        i,
                    )

                    self.forced_adapter.set_check_matrix_and_priors(
                        constrained.check_matrix,
                        constrained.priors.copy(),
                    )
                    result_i = self.forced_adapter.decode_detailed_single(constrained.syndrome)
                    if bool(result_i.success):
                        c_i = np.asarray(result_i.decoding, dtype=np.bool_)[
                            : constrained.physical_cols
                        ]
                        l_i = _logical_from_correction(self._observables, c_i)
                        w_i = float(self._weights @ c_i.astype(int))
                        forced_results.append((w_i, l_i, float(result_i.iterations)))
                        all_solutions.append((w_i, l_i))

                # Restore baseline state for the next shot's Phase 1.
                self.forced_adapter.set_check_matrix_and_priors(
                    self._base_check_matrix,
                    self._base_priors.copy(),
                )

            # --- Phase 3: gap & decision ---
            # Sort by weight; stable sort preserves the baseline's priority on ties.
            all_solutions.sort(key=lambda x: x[0])
            ml_weight, ml_logical = all_solutions[0]
            predictions[shot] = ml_logical.astype(np.bool_)

            best_diff_weight = np.inf
            for w, l in all_solutions[1:]:
                if not np.array_equal(l, ml_logical):
                    best_diff_weight = w
                    break

            if len(all_solutions) == 1:
                # All K forced runs failed to converge (fixed paper
                # convention, not configurable): Gap = +inf, always accepted.
                gap[shot] = np.inf
            else:
                gap[shot] = best_diff_weight - ml_weight

            diff = np.asarray(l0, dtype=int) ^ np.asarray(ml_logical, dtype=int)
            obs_flip_idx.append(list(np.where(diff)[0]))

            if get_detail_stat:
                num_forced_converged[shot] = len(forced_results)
                if forced_results:
                    forced_results.sort(key=lambda x: x[0])
                    forced_iteration[shot] = forced_results[0][2]
                    if len(forced_results) >= 2:
                        forced_2nd_best_iteration[shot] = forced_results[1][2]

        metrics: dict[str, Any] = {
            WILLS_REPRODUCE_METRIC: gap,
            "__is_logical_error": force_logical_error,
        }
        decoder_stats: dict[str, Any] = {}
        if get_detail_stat:
            decoder_stats = {
                "baseline_iteration": baseline_iteration,
                "forced_iteration": forced_iteration,
                "forced_2nd_best_iteration": forced_2nd_best_iteration,
                "num_forced_converged": num_forced_converged,
            }

        return DecodingResult(
            predictions=predictions,
            metrics=metrics,
            detail_stats={},
            decoder_stats=decoder_stats,
            obs_flip_idx=obs_flip_idx,
        )


@dataclass(frozen=True)
class _WillsReproduceFactory:
    dem_path: Path
    decoder_name: str
    decoder_options: Mapping[str, Any]
    metric_options: Mapping[str, Any]

    def __call__(self, dem: stim.DetectorErrorModel | None = None) -> WillsReproduceDecoder:
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(str(self.dem_path))

        options = _parse_wills_reproduce_options(self.metric_options)

        baseline_adapter = build_decoder_adapter(self.decoder_name, dem, self.decoder_options)
        if not isinstance(baseline_adapter, RelayBpDecoderAdapter):
            raise ValueError(
                "wills_reproduce reproduces a Relay-BP-specific post-selection "
                "strategy (Wills, Yoder & Chuang, arXiv:2605.20346); "
                f"got decoder={self.decoder_name!r}"
            )

        forced_decoder_options = dict(self.decoder_options)
        forced_decoder_options["num_sets"] = options.forced_num_sets
        forced_adapter = build_decoder_adapter(self.decoder_name, dem, forced_decoder_options)
        assert isinstance(forced_adapter, RelayBpDecoderAdapter)

        return WillsReproduceDecoder(
            baseline_adapter=baseline_adapter,
            forced_adapter=forced_adapter,
            options=options,
        )


def make_wills_reproduce_factory(
    dem_path: Path,
    decoder_name: str,
    decoder_options: Mapping[str, Any],
    metric_options: Mapping[str, Any],
) -> DecoderFactory:
    return _WillsReproduceFactory(
        dem_path=dem_path,
        decoder_name=decoder_name,
        decoder_options=dict(decoder_options),
        metric_options=dict(metric_options),
    )
