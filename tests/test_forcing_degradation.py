from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from analysis.src.forcing_degradation import (
    ForcingDegradationConfig,
    ForcingPlotOptions,
    descriptive_statistics,
    load_forcing_degradation_lazy,
    plot_metric_frequency,
)
from decoder_confidence.decoding.forcing_degradation_collect import main as forcing_main
from decoder_confidence.decoding._forcing_degradation_test import (
    BpLsdForcingStageRunner,
    ForcingDegradationTestDecoder,
    RelayBpForcingStageRunner,
    StageDecode,
    _append_observable_constraints,
)
from decoder_confidence.sampling.__main__ import main as sampling_main

REPO_ROOT = Path(__file__).resolve().parents[1]
from decoder_confidence.decoding.forcing_degradation_collect import (
    _collect_forcing_degradation_results,
)


class _FakeRunner:
    metric_name = "iteration"
    observables_matrix = np.array([[1, 0, 1]], dtype=np.uint8)

    def __init__(self, *, forced_logical: np.ndarray | None = None) -> None:
        self.forced_logical = (
            np.array([1], dtype=np.bool_)
            if forced_logical is None
            else forced_logical
        )
        self.force_nonconverged = False
        self.forced_target: np.ndarray | None = None
        self.reset_called = False

    def decode_baseline(self, syndrome: np.ndarray) -> StageDecode:
        return StageDecode(
            correction=np.array([1, 0, 0], dtype=np.bool_),
            logical_class=np.array([1], dtype=np.bool_),
            weight=2.0,
            metric=3,
        )

    def decode_forced(
        self,
        syndrome: np.ndarray,
        baseline_logical_class: np.ndarray,
    ) -> StageDecode:
        self.forced_target = baseline_logical_class.copy()
        return StageDecode(
            correction=np.array([0, 1, 0], dtype=np.bool_),
            logical_class=self.forced_logical,
            weight=5.0,
            metric=9,
            converged=not self.force_nonconverged,
        )

    def reset(self) -> None:
        self.reset_called = True


def test_forcing_decoder_records_baseline_outputs_and_weights() -> None:
    runner = _FakeRunner()
    decoder = ForcingDegradationTestDecoder(runner=runner)

    result = decoder.decode(np.array([[0, 1]], dtype=np.uint8))

    assert result.predictions.tolist() == [[True]]
    assert runner.forced_target is not None
    assert runner.forced_target.tolist() == [True]
    assert runner.reset_called
    assert result.metrics["baseline_weight"].tolist() == [2.0]
    assert result.metrics["forced_weight"].tolist() == [5.0]
    assert result.metrics["baseline_iteration"].tolist() == [3]
    assert result.metrics["forced_iteration"].tolist() == [9]

    sampled_obs = np.array([[0]], dtype=np.bool_)
    logicalerror = (result.predictions ^ sampled_obs).any(axis=1)
    assert logicalerror.tolist() == [True]


def test_forcing_decoder_rejects_inconsistent_forced_logical_class() -> None:
    decoder = ForcingDegradationTestDecoder(
        runner=_FakeRunner(forced_logical=np.array([0], dtype=np.bool_))
    )

    with pytest.raises(RuntimeError, match="Forced run logical class"):
        decoder.decode(np.array([[0, 1]], dtype=np.uint8))


def test_forcing_decoder_skips_logical_class_check_when_nonconverged() -> None:
    runner = _FakeRunner(forced_logical=np.array([0], dtype=np.bool_))
    runner.force_nonconverged = True
    decoder = ForcingDegradationTestDecoder(runner=runner)

    result = decoder.decode(np.array([[0, 1]], dtype=np.uint8))

    assert result.predictions.tolist() == [[True]]
    assert result.metrics["__is_logical_error"].tolist() == [True]
    assert result.metrics["forced_iteration"].tolist() == [9]
    assert result.metrics["forced_weight"].tolist() == [5.0]


def test_append_observable_constraints_dense() -> None:
    check = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    obs = np.array([[1, 1, 0]], dtype=np.uint8)

    augmented = _append_observable_constraints(check, obs)

    assert augmented.tolist() == [[1, 0, 1], [0, 1, 1], [1, 1, 0]]


@dataclass
class _DetailedResult:
    decoding: np.ndarray
    iterations: int
    success: bool = True


class _FakeRelayAdapter:
    def __init__(self) -> None:
        self.check_matrix = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        self.observables_matrix = np.array([[1, 1]], dtype=np.uint8)
        self.priors = np.array([0.2, 0.3], dtype=float)
        self._results = [
            _DetailedResult(np.array([1, 0], dtype=np.uint8), 4),
            _DetailedResult(np.array([0, 1], dtype=np.uint8), 30, success=False),
        ]
        self.seen_syndromes: list[list[int]] = []

    def decode_detailed_single(self, syndrome: np.ndarray) -> _DetailedResult:
        self.seen_syndromes.append(np.asarray(syndrome, dtype=int).tolist())
        return self._results.pop(0)

    def set_check_matrix(self, check_matrix) -> None:
        self.check_matrix = check_matrix

    def set_priors(self, priors: np.ndarray) -> None:
        self.priors = priors


def test_relay_runner_records_iterations_and_forced_syndrome() -> None:
    adapter = _FakeRelayAdapter()
    runner = RelayBpForcingStageRunner(adapter=adapter)  # type: ignore[arg-type]

    baseline = runner.decode_baseline(np.array([1, 0], dtype=np.uint8))
    forced = runner.decode_forced(
        np.array([1, 0], dtype=np.uint8),
        baseline.logical_class,
    )

    assert baseline.metric == 4
    assert forced.metric == 30
    assert baseline.converged is True
    assert forced.converged is False
    assert adapter.seen_syndromes == [[1, 0], [1, 0, 1]]


def test_bplsd_runner_always_enables_stats_and_computes_cluster_llr(monkeypatch) -> None:
    constructed: list["_FakeBpLsdDecoder"] = []

    class _FakeBpLsdDecoder:
        corrections = [
            np.array([1, 0], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
        ]

        def __init__(self, check_matrix, error_channel, **options) -> None:
            self.check_matrix = check_matrix
            self.error_channel = error_channel
            self.options = options
            self.do_stats = False
            self.statistics = {
                "individual_cluster_stats": {
                    0: {
                        "active": True,
                        "absorbed_by_cluster": -1,
                        "final_bits": [0],
                    }
                }
            }
            constructed.append(self)

        def set_do_stats(self, value: bool) -> None:
            self.do_stats = value

        def decode(self, syndrome: np.ndarray) -> np.ndarray:
            return self.corrections.pop(0)

    ldpc_mod = types.ModuleType("ldpc")
    bplsd_mod = types.ModuleType("ldpc.bplsd_decoder")
    bplsd_mod.BpLsdDecoder = _FakeBpLsdDecoder
    monkeypatch.setitem(sys.modules, "ldpc", ldpc_mod)
    monkeypatch.setitem(sys.modules, "ldpc.bplsd_decoder", bplsd_mod)

    runner = BpLsdForcingStageRunner(
        check_matrix=np.array([[1, 0], [0, 1]], dtype=np.uint8),
        observables_matrix=np.array([[1, 1]], dtype=np.uint8),
        priors=np.array([0.2, 0.3], dtype=float),
        decoder_options={"max_iter": 1},
        alpha=2.0,
    )

    baseline = runner.decode_baseline(np.array([1, 0], dtype=np.uint8))
    forced = runner.decode_forced(
        np.array([1, 0], dtype=np.uint8),
        baseline.logical_class,
    )

    assert baseline.metric > 0.0
    assert forced.metric > 0.0
    assert len(constructed) >= 3
    assert all(decoder.options["always_run_lsd"] is True for decoder in constructed)
    assert all(decoder.do_stats for decoder in constructed)


def test_collect_forcing_degradation_results_writes_expected_parquets(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunks" / "batch=1"
    output_dir = tmp_path / "decoding_result" / "decoder=BP-LSD,metric=forcing_degradation_test"
    chunk_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    pl.DataFrame(
        {
            "shot_id": np.array([0, 1], dtype=np.int64),
            "is_logical_error": [False, True],
            "metric_baseline_weight": np.array([1.0, 2.0], dtype=np.float64),
            "metric_forced_weight": np.array([1.5, 3.0], dtype=np.float64),
            "metric_baseline_cluster_llr": np.array([0.1, 0.2], dtype=np.float64),
            "metric_forced_cluster_llr": np.array([0.3, 0.4], dtype=np.float64),
        }
    ).write_parquet(chunk_dir / "chunk_000001_deadbeef.parquet")

    columns = _collect_forcing_degradation_results(chunk_dir, output_dir, 1)

    assert columns == [
        "logicalerror",
        "baseline_weight",
        "forced_weight",
        "baseline_cluster_llr",
        "forced_cluster_llr",
    ]
    metric_path = output_dir / "metric=forcing_degradation_test_batch=1.parquet"
    logical_path = output_dir / "logicalerror_batch=1.parquet"
    assert metric_path.exists()
    assert logical_path.exists()

    df = pl.read_parquet(metric_path)
    assert df.columns == ["shot_id", *columns]
    assert df["logicalerror"].to_list() == [False, True]
    assert df["baseline_weight"].to_list() == [1.0, 2.0]
    assert df["forced_weight"].to_list() == [1.5, 3.0]
    assert df.schema["logicalerror"] == pl.Boolean
    assert df.schema["baseline_cluster_llr"] == pl.Float64

    le_df = pl.read_parquet(logical_path)
    assert le_df.columns == ["shot_id", "is_logical_error"]
    assert le_df["is_logical_error"].to_list() == [False, True]


def test_forcing_degradation_analysis_load_stats_and_plot(tmp_path: Path) -> None:
    result_dir = (
        tmp_path
        / "code=surface_code_Z,d=5,rounds=5,noisemodel=si1000,p=0.001,xyz=False"
        / "decoding_result"
        / "decoder=RELAY-BP,metric=forcing_degradation_test"
    )
    result_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "shot_id": [0, 1, 2],
            "logicalerror": [False, True, False],
            "baseline_weight": [1.0, 2.0, 1.0],
            "forced_weight": [1.5, 3.0, 1.5],
            "baseline_iteration": [2, 5, 2],
            "forced_iteration": [4, 6, 4],
        }
    ).write_parquet(result_dir / "metric=forcing_degradation_test_batch=1.parquet")

    lf = load_forcing_degradation_lazy(
        tmp_path,
        ForcingDegradationConfig(
            filters={"d": 5, "p": 0.001},
            decoder_names=["RELAY-BP"],
        ),
    )
    df = lf.collect()

    assert df.height == 3
    assert set(["decoder", "batch"]).issubset(df.columns)

    stats = descriptive_statistics(lf, separate_logicalerror=True)
    assert {"logicalerror", "column", "count", "mean", "std", "min", "max"}.issubset(
        set(stats.columns)
    )
    assert "baseline_iteration" in stats["column"].to_list()
    assert "forced_iteration" in stats["column"].to_list()

    fig, ax = plot_metric_frequency(
        df,
        stage="baseline",
        options=ForcingPlotOptions(
            use_latex=False,
            distinguish_logicalerror=True,
            show_legend=True,
        ),
    )
    assert ax.get_xlabel() == "baseline_iteration"
    assert len(ax.collections) == 2
    fig.clf()


def _sample_forcing_data(tmp_path: Path, *, num_batch: int) -> Path:
    out_dir = tmp_path / "out"
    rc = sampling_main(
        [
            "--code",
            "surface_code_Z",
            "--out_dir",
            str(out_dir),
            "--noise_model",
            "si1000",
            "--rounds",
            "5",
            "--d",
            "5",
            "--p",
            "0.001",
            "--num_shots",
            "4",
            "--det_sample_seed",
            "0",
            "--num_batch",
            str(num_batch),
            "--xyz_decoding",
            "False",
        ]
    )
    assert rc == 0
    return out_dir / "code=surface_code_Z,d=5,rounds=5,noisemodel=si1000,p=0.001,xyz=False"


def _forcing_args(
    data_dir: Path,
    config_path: Path,
    *,
    batch_num: int,
    num_workers: int,
) -> list[str]:
    return [
        "--code",
        "surface_code_Z",
        "--data_dir",
        str(data_dir),
        "--noise_model",
        "si1000",
        "--rounds",
        "5",
        "--d",
        "5",
        "--p",
        "0.001",
        "--batch_num",
        str(batch_num),
        "--num_workers",
        str(num_workers),
        "--decoder_config",
        str(config_path),
        "--xyz",
        "False",
    ]


@pytest.mark.e2e
def test_forcing_degradation_bplsd_e2e_batch_parallel(tmp_path: Path) -> None:
    pytest.importorskip("ldpc.bplsd_decoder")

    circuit_dir = _sample_forcing_data(tmp_path, num_batch=2)
    data_dir = circuit_dir.parent
    config_path = REPO_ROOT / "conf" / "forcing_degradation_bplsd.yaml"

    assert forcing_main(_forcing_args(data_dir, config_path, batch_num=1, num_workers=2)) == 0
    assert forcing_main(_forcing_args(data_dir, config_path, batch_num=2, num_workers=2)) == 0

    output_dir = (
        circuit_dir
        / "decoding_result"
        / "decoder=BP-LSD,metric=forcing_degradation_test,alpha=2.0"
    )
    for batch_num in (1, 2):
        df = pl.read_parquet(
            output_dir / f"metric=forcing_degradation_test_batch={batch_num}.parquet"
        )
        assert df.columns == [
            "shot_id",
            "logicalerror",
            "baseline_weight",
            "forced_weight",
            "baseline_cluster_llr",
            "forced_cluster_llr",
        ]
        assert df.schema["logicalerror"] == pl.Boolean
        assert (output_dir / f"logicalerror_batch={batch_num}.parquet").exists()


@pytest.mark.e2e
def test_forcing_degradation_relay_bp_e2e(tmp_path: Path) -> None:
    pytest.importorskip("relay_bp")

    circuit_dir = _sample_forcing_data(tmp_path, num_batch=1)
    data_dir = circuit_dir.parent
    config_path = REPO_ROOT / "conf" / "forcing_degradation_relay_bp.yaml"

    assert forcing_main(_forcing_args(data_dir, config_path, batch_num=1, num_workers=1)) == 0

    output_dir = (
        circuit_dir
        / "decoding_result"
        / "decoder=RELAY-BP,metric=forcing_degradation_test"
    )
    df = pl.read_parquet(output_dir / "metric=forcing_degradation_test_batch=1.parquet")
    assert df.columns == [
        "shot_id",
        "logicalerror",
        "baseline_weight",
        "forced_weight",
        "baseline_iteration",
        "forced_iteration",
    ]
    assert df.schema["baseline_iteration"] == pl.Int64
