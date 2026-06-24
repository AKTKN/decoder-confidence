from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoder_confidence.config import SamplingConfig
from decoder_confidence.decoding._constraints import (  # noqa: E402
    ConstrainedDecodeOptions,
    _append_row,
    _split_check_matrix,
    build_partition,
)
from decoder_confidence.decoding._decoder_adapter import (  # noqa: E402
    DemMatricesWithEdges,
    _dem_to_matrices_with_edge_priors,
)
from decoder_confidence.sampling.dem import (  # noqa: E402
    filter_dem_by_basis,
    find_circuit_file,
    load_circuit,
)


@dataclass(frozen=True)
class QuantumCircuitParameters:
    """Parameters used to locate and generate the circuit-level DEM."""

    code: str
    d: int
    rounds: str | int
    noise_model: str
    p: float
    xyz_decoding: bool = False
    circuits_dir: Path = REPO_ROOT / "circuits"

    def to_sampling_config(self) -> SamplingConfig:
        """Return a minimal SamplingConfig compatible with step-2 helpers."""
        return SamplingConfig(
            code=self.code,
            out_dir=Path("."),
            noise_model=self.noise_model,
            rounds=str(self.rounds),
            d=int(self.d),
            p=float(self.p),
            num_shots=1,
            det_sample_seed=0,
            num_batch=1,
            xyz_decoding=bool(self.xyz_decoding),
        )


@dataclass(frozen=True)
class ObservableWeightResult:
    """Support weight for one logical-observable row."""

    observable_index: int
    support_size: int


@dataclass(frozen=True)
class Length4CycleResult:
    """Cycle count for one unsplit logical-observable constraint."""

    observable_index: int
    cycle_count: int
    support_size: int


@dataclass(frozen=True)
class RandomSplitCycleResult:
    """Cycle count for one randomly split logical-observable constraint."""

    observable_index: int
    cycle_count: int
    support_size: int
    requested_splits: int
    actual_splits: int
    split_seed: int
    split_balanced: bool
    block_sizes: tuple[int, ...]


@dataclass(frozen=True)
class MonteCarloCycleResult:
    """Monte Carlo summary for random split cycle counts."""

    observable_index: int
    sample_count: int
    expected_cycle_count: float
    std_cycle_count: float
    min_cycle_count: int
    max_cycle_count: int
    cycle_counts: tuple[int, ...]
    seeds: tuple[int, ...]


def _needs_dem_filter(code: str, xyz_decoding: bool) -> bool:
    if xyz_decoding:
        return False
    return code.startswith("surface_code")


def _get_remove_basis(code: str) -> str:
    if code.endswith("_Z"):
        return "X"
    if code.endswith("_X"):
        return "Z"
    raise ValueError(
        f"Cannot infer remove_basis from code name {code!r}. "
        "Expected a code name ending in '_Z' or '_X'."
    )


def _row_to_dense(row: Any) -> np.ndarray:
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    if sparse is not None and sparse.issparse(row):
        return np.asarray(row.todense(), dtype=np.uint8).ravel()
    return np.asarray(row, dtype=np.uint8).ravel()


def _as_csr_binary(matrix: Any):
    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - scipy is required upstream too
        raise RuntimeError("scipy is required for length-4 cycle analysis") from exc

    if sparse.issparse(matrix):
        return matrix.astype(np.uint8).tocsr()
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.uint8))


def _comb2(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.int64, copy=False)
    return values64 * (values64 - 1) // 2


def _results_to_cycle_map(results: Iterable[Any], value_attr: str) -> dict[int, float]:
    return {
        int(result.observable_index): float(getattr(result, value_attr))
        for result in results
    }


class Length4CycleAnalyzer:
    """Analyze new length-4 cycles induced by logical parity constraints.

    The analyzer uses the same circuit lookup and DEM-to-matrix conversion path
    as the decoding pipeline. Cycle counts are computed against the original
    parity-check rows only; random split auxiliary columns therefore do not add
    overlap with original checks.
    """

    def __init__(
        self,
        parameters: QuantumCircuitParameters,
        *,
        allow_undecomposed_hyperedges: bool = True,
    ) -> None:
        self.parameters = parameters
        self.allow_undecomposed_hyperedges = bool(allow_undecomposed_hyperedges)
        self.circuit_path: Path | None = None
        self.circuit: Any | None = None
        self.dem: Any | None = None
        self.matrices: DemMatricesWithEdges | None = None
        self._check_matrix_csr: Any | None = None

    @classmethod
    def from_parameters(
        cls,
        *,
        code: str,
        d: int,
        rounds: str | int,
        noise_model: str | None = None,
        noisemodel: str | None = None,
        p: float,
        xyz_decoding: bool = False,
        circuits_dir: str | Path | None = None,
        allow_undecomposed_hyperedges: bool = True,
    ) -> "Length4CycleAnalyzer":
        """Construct an analyzer from notebook-friendly keyword arguments."""
        model = noise_model if noise_model is not None else noisemodel
        if model is None:
            raise ValueError("noise_model or noisemodel must be provided")
        params = QuantumCircuitParameters(
            code=code,
            d=d,
            rounds=rounds,
            noise_model=model,
            p=p,
            xyz_decoding=xyz_decoding,
            circuits_dir=Path(circuits_dir) if circuits_dir is not None else REPO_ROOT / "circuits",
        )
        return cls(params, allow_undecomposed_hyperedges=allow_undecomposed_hyperedges)

    @property
    def check_matrix(self) -> Any:
        self.load_matrices()
        assert self.matrices is not None
        return self.matrices.check_matrix

    @property
    def observable_matrix(self) -> Any:
        self.load_matrices()
        assert self.matrices is not None
        return self.matrices.observables_matrix

    def load_circuit(self) -> Any:
        """Load the Stim circuit matching the requested parameters."""
        if self.circuit is None:
            config = self.parameters.to_sampling_config()
            self.circuit_path = find_circuit_file(self.parameters.circuits_dir, config)
            self.circuit = load_circuit(self.circuit_path)
        return self.circuit

    def build_detector_error_model(self) -> Any:
        """Build the DEM using the same filtering policy as step 2."""
        if self.dem is None:
            circuit = self.load_circuit()
            dem = circuit.detector_error_model(decompose_errors=False)
            if _needs_dem_filter(self.parameters.code, self.parameters.xyz_decoding):
                dem = filter_dem_by_basis(dem, _get_remove_basis(self.parameters.code))
            self.dem = dem
        return self.dem

    def load_matrices(self) -> DemMatricesWithEdges:
        """Convert the DEM into check and logical-observable matrices."""
        if self.matrices is None:
            dem = self.build_detector_error_model()
            self.matrices = _dem_to_matrices_with_edge_priors(
                dem,
                allow_undecomposed_hyperedges=self.allow_undecomposed_hyperedges,
            )
            self._check_matrix_csr = _as_csr_binary(self.matrices.check_matrix)
        return self.matrices

    def observable_count(self) -> int:
        """Return the number of logical observable rows."""
        return int(self.observable_matrix.shape[0])

    def observable_row(self, observable_index: int) -> np.ndarray:
        """Return one logical-observable row as a dense uint8 vector."""
        matrix = self.observable_matrix
        n_obs = int(matrix.shape[0])
        if observable_index < 0 or observable_index >= n_obs:
            raise IndexError(
                f"observable_index={observable_index} is out of range for {n_obs} observables"
            )
        return _row_to_dense(matrix[observable_index])

    def observable_support(self, observable_index: int) -> np.ndarray:
        """Return physical error-mechanism columns touched by an observable."""
        row = self.observable_row(observable_index)
        return np.flatnonzero(row).astype(np.int64)

    def observable_weight(self, observable_index: int) -> ObservableWeightResult:
        """Return the Hamming weight of one logical-observable row."""
        support = self.observable_support(observable_index)
        return ObservableWeightResult(
            observable_index=int(observable_index),
            support_size=int(support.size),
        )

    def observable_weights(self) -> list[ObservableWeightResult]:
        """Return logical-observable row weights for all observables."""
        return [self.observable_weight(i) for i in range(self.observable_count())]

    def create_extended_check_matrix(
        self,
        observable_index: int,
        *,
        random_split: bool = False,
        n_splits: int = 3,
        split_seed: int = 0,
        split_balanced: bool = False,
    ) -> Any:
        """Return the parity-check matrix with the requested observable constraint."""
        row = self.observable_row(observable_index)
        if not random_split:
            return _append_row(self.check_matrix, row)

        partition = build_partition(
            row,
            ConstrainedDecodeOptions(
                random_split=True,
                n_splits=n_splits,
                split_seed=split_seed,
                split_balanced=split_balanced,
            ),
        )
        if partition is None:
            return _append_row(self.check_matrix, row)
        return _split_check_matrix(self.check_matrix, int(self.check_matrix.shape[1]), partition)

    def count_cycles_for_support(self, support: Iterable[int]) -> int:
        """Count new cycles induced by one added check over physical columns."""
        support_arr = np.fromiter((int(x) for x in support), dtype=np.int64)
        if support_arr.size < 2:
            return 0
        self.load_matrices()
        assert self._check_matrix_csr is not None
        overlaps = np.asarray(self._check_matrix_csr[:, support_arr].getnnz(axis=1)).ravel()
        return int(_comb2(overlaps).sum())

    def overlap_counts_for_support(self, support: Iterable[int]) -> np.ndarray:
        """Return overlap counts between each original check row and a support set."""
        support_arr = np.fromiter((int(x) for x in support), dtype=np.int64)
        self.load_matrices()
        assert self._check_matrix_csr is not None
        if support_arr.size == 0:
            return np.zeros(int(self._check_matrix_csr.shape[0]), dtype=np.int64)
        return np.asarray(self._check_matrix_csr[:, support_arr].getnnz(axis=1)).ravel()

    def overlap_summary(self, observable_index: int) -> dict[str, Any]:
        """Summarize check-row overlaps for one logical observable."""
        support = self.observable_support(observable_index)
        overlaps = self.overlap_counts_for_support(support)
        values, counts = np.unique(overlaps, return_counts=True)
        return {
            "observable_index": int(observable_index),
            "support_size": int(support.size),
            "max_overlap": int(overlaps.max(initial=0)),
            "checks_with_overlap_ge_2": int(np.count_nonzero(overlaps >= 2)),
            "new_length4_cycles": int(_comb2(overlaps).sum()),
            "overlap_distribution": {
                int(value): int(count) for value, count in zip(values, counts)
            },
        }

    def analyze_observable(self, observable_index: int) -> Length4CycleResult:
        """Analyze the unsplit constraint for one logical observable."""
        support = self.observable_support(observable_index)
        return Length4CycleResult(
            observable_index=int(observable_index),
            cycle_count=self.count_cycles_for_support(support),
            support_size=int(support.size),
        )

    def analyze_all_observables(self) -> list[Length4CycleResult]:
        """Analyze the unsplit constraint for every logical observable."""
        return [self.analyze_observable(i) for i in range(self.observable_count())]

    def analyze_random_split(
        self,
        observable_index: int,
        *,
        n_splits: int = 3,
        split_seed: int = 0,
        split_balanced: bool = False,
    ) -> RandomSplitCycleResult:
        """Analyze one random split of a logical-observable constraint."""
        row = self.observable_row(observable_index)
        support = np.flatnonzero(row).astype(np.int64)
        partition = build_partition(
            row,
            ConstrainedDecodeOptions(
                random_split=True,
                n_splits=n_splits,
                split_seed=split_seed,
                split_balanced=split_balanced,
            ),
        )
        if partition is None:
            cycle_count = self.count_cycles_for_support(support)
            block_sizes = (int(support.size),)
            actual_splits = 1
        else:
            cycle_count = sum(self.count_cycles_for_support(block) for block in partition.blocks)
            block_sizes = tuple(len(block) for block in partition.blocks)
            actual_splits = len(partition.blocks)

        return RandomSplitCycleResult(
            observable_index=int(observable_index),
            cycle_count=int(cycle_count),
            support_size=int(support.size),
            requested_splits=int(n_splits),
            actual_splits=int(actual_splits),
            split_seed=int(split_seed),
            split_balanced=bool(split_balanced),
            block_sizes=block_sizes,
        )

    def monte_carlo_random_split(
        self,
        observable_index: int,
        *,
        n_splits: int = 3,
        sample_count: int = 100,
        seed: int = 0,
        split_balanced: bool = False,
    ) -> MonteCarloCycleResult:
        """Estimate the expected new cycle count over random split seeds."""
        if sample_count < 1:
            raise ValueError(f"sample_count must be >= 1 but got {sample_count}")

        rng = np.random.default_rng(int(seed))
        seeds_arr = rng.integers(0, np.iinfo(np.int64).max, size=int(sample_count), dtype=np.int64)
        counts: list[int] = []
        seeds: list[int] = []
        for split_seed in seeds_arr:
            seed_int = int(split_seed)
            result = self.analyze_random_split(
                observable_index,
                n_splits=n_splits,
                split_seed=seed_int,
                split_balanced=split_balanced,
            )
            counts.append(int(result.cycle_count))
            seeds.append(seed_int)

        values = np.asarray(counts, dtype=float)
        return MonteCarloCycleResult(
            observable_index=int(observable_index),
            sample_count=int(sample_count),
            expected_cycle_count=float(values.mean()),
            std_cycle_count=float(values.std(ddof=0)),
            min_cycle_count=int(values.min()),
            max_cycle_count=int(values.max()),
            cycle_counts=tuple(counts),
            seeds=tuple(seeds),
        )

    def monte_carlo_all_observables(
        self,
        *,
        n_splits: int = 3,
        sample_count: int = 100,
        seed: int = 0,
        split_balanced: bool = False,
    ) -> list[MonteCarloCycleResult]:
        """Estimate random split cycle counts for every logical observable."""
        return [
            self.monte_carlo_random_split(
                i,
                n_splits=n_splits,
                sample_count=sample_count,
                seed=seed + i,
                split_balanced=split_balanced,
            )
            for i in range(self.observable_count())
        ]

    def build_cycle_comparison_table(
        self,
        *,
        n_splits: int = 3,
        split_seed: int = 0,
        split_balanced: bool = False,
    ) -> list[dict[str, Any]]:
        """Return unsplit and one random-split cycle count per observable."""
        unsplit = self.analyze_all_observables()
        split = [
            self.analyze_random_split(
                i,
                n_splits=n_splits,
                split_seed=split_seed,
                split_balanced=split_balanced,
            )
            for i in range(self.observable_count())
        ]
        split_by_obs = {result.observable_index: result for result in split}
        rows: list[dict[str, Any]] = []
        for result in unsplit:
            split_result = split_by_obs[result.observable_index]
            rows.append(
                {
                    "observable_index": result.observable_index,
                    "support_size": result.support_size,
                    "unsplit_new_length4_cycles": result.cycle_count,
                    "split_new_length4_cycles": split_result.cycle_count,
                    "requested_splits": split_result.requested_splits,
                    "actual_splits": split_result.actual_splits,
                    "block_sizes": split_result.block_sizes,
                    "split_seed": split_result.split_seed,
                    "split_balanced": split_result.split_balanced,
                }
            )
        return rows

    def build_monte_carlo_comparison_table(
        self,
        *,
        n_splits: int = 3,
        sample_count: int = 100,
        seed: int = 0,
        split_balanced: bool = False,
    ) -> list[dict[str, Any]]:
        """Return unsplit/m and Monte Carlo random-split expected cycle counts."""
        unsplit = self.analyze_all_observables()
        mc = self.monte_carlo_all_observables(
            n_splits=n_splits,
            sample_count=sample_count,
            seed=seed,
            split_balanced=split_balanced,
        )
        mc_by_obs = {result.observable_index: result for result in mc}
        rows: list[dict[str, Any]] = []
        for result in unsplit:
            mc_result = mc_by_obs[result.observable_index]
            rows.append(
                {
                    "observable_index": result.observable_index,
                    "support_size": result.support_size,
                    "unsplit_new_length4_cycles": result.cycle_count,
                    "unsplit_divided_by_m": result.cycle_count / float(n_splits),
                    "mc_expected_split_new_length4_cycles": mc_result.expected_cycle_count,
                    "mc_std_split_new_length4_cycles": mc_result.std_cycle_count,
                    "mc_min_split_new_length4_cycles": mc_result.min_cycle_count,
                    "mc_max_split_new_length4_cycles": mc_result.max_cycle_count,
                    "n_splits": int(n_splits),
                    "sample_count": mc_result.sample_count,
                    "seed": int(seed),
                    "split_balanced": bool(split_balanced),
                }
            )
        return rows

    def plot_unsplit_vs_split(
        self,
        *,
        n_splits: int = 3,
        split_seed: int = 0,
        split_balanced: bool = False,
        ax: Any | None = None,
        figsize: tuple[float, float] = (8.0, 4.0),
        bar_width: float = 0.38,
        colors: tuple[str, str] = ("#4C78A8", "#F58518"),
        labels: tuple[str, str] = (
            r"Unsplit",
            r"Random split",
        ),
        xlabel: str = r"Observable index",
        ylabel: str = r"New length-4 cycles",
        title: str | None = None,
        use_tex: bool = False,
        font_size: float = 11.0,
        legend: bool = True,
        grid: bool = True,
    ) -> Any:
        """Plot unsplit vs one random-split cycle count as grouped bars."""
        import matplotlib.pyplot as plt

        table = self.build_cycle_comparison_table(
            n_splits=n_splits,
            split_seed=split_seed,
            split_balanced=split_balanced,
        )
        x = np.asarray([row["observable_index"] for row in table], dtype=float)
        unsplit = np.asarray([row["unsplit_new_length4_cycles"] for row in table], dtype=float)
        split = np.asarray([row["split_new_length4_cycles"] for row in table], dtype=float)

        with plt.rc_context({"text.usetex": use_tex, "font.size": font_size}):
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            ax.bar(x - bar_width / 2, unsplit, width=bar_width, color=colors[0], label=labels[0])
            ax.bar(x + bar_width / 2, split, width=bar_width, color=colors[1], label=labels[1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if title is not None:
                ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) for v in x])
            if grid:
                ax.grid(axis="y", alpha=0.25)
                ax.set_axisbelow(True)
            if legend:
                ax.legend(frameon=False)
        return ax

    def plot_unsplit_div_m_vs_monte_carlo(
        self,
        *,
        n_splits: int = 3,
        sample_count: int = 100,
        seed: int = 0,
        split_balanced: bool = False,
        ax: Any | None = None,
        figsize: tuple[float, float] = (8.0, 4.0),
        bar_width: float = 0.38,
        colors: tuple[str, str] = ("#54A24B", "#E45756"),
        labels: tuple[str, str] | None = None,
        xlabel: str = r"Observable index",
        ylabel: str = r"New length-4 cycles",
        title: str | None = None,
        use_tex: bool = False,
        font_size: float = 11.0,
        legend: bool = True,
        grid: bool = True,
        show_errorbar: bool = True,
        errorbar_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Plot unsplit/m against Monte Carlo expected random-split cycle counts."""
        import matplotlib.pyplot as plt

        if labels is None:
            labels = (rf"Unsplit / {int(n_splits)}", r"MC split expectation")
        table = self.build_monte_carlo_comparison_table(
            n_splits=n_splits,
            sample_count=sample_count,
            seed=seed,
            split_balanced=split_balanced,
        )
        x = np.asarray([row["observable_index"] for row in table], dtype=float)
        scaled = np.asarray([row["unsplit_divided_by_m"] for row in table], dtype=float)
        expected = np.asarray(
            [row["mc_expected_split_new_length4_cycles"] for row in table], dtype=float
        )
        std = np.asarray([row["mc_std_split_new_length4_cycles"] for row in table], dtype=float)
        err_kwargs = {"fmt": "none", "ecolor": "#222222", "elinewidth": 1.0, "capsize": 2.5}
        if errorbar_kwargs:
            err_kwargs.update(errorbar_kwargs)

        with plt.rc_context({"text.usetex": use_tex, "font.size": font_size}):
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            ax.bar(x - bar_width / 2, scaled, width=bar_width, color=colors[0], label=labels[0])
            ax.bar(x + bar_width / 2, expected, width=bar_width, color=colors[1], label=labels[1])
            if show_errorbar:
                ax.errorbar(x + bar_width / 2, expected, yerr=std, **err_kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if title is not None:
                ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) for v in x])
            if grid:
                ax.grid(axis="y", alpha=0.25)
                ax.set_axisbelow(True)
            if legend:
                ax.legend(frameon=False)
        return ax
