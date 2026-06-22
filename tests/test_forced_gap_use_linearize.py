from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis.src.config import PlotConfig
from analysis.src.data_manager import SimulationDataManager
from analysis.src.postselect import PostSelectSpec, PostSelectionPlotter


def _write_batch(dm_dir: Path, metric_name: str, metric_values: list[float], errors: list[bool]) -> None:
    dm_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "shot_id": [0, 1],
            metric_name: metric_values,
        }
    ).write_parquet(dm_dir / f"metric={metric_name}_batch=1.parquet")
    pl.DataFrame(
        {
            "shot_id": [0, 1],
            "is_logical_error": errors,
        }
    ).write_parquet(dm_dir / "logicalerror_batch=1.parquet")


def test_query_forced_gap_can_use_linearize_logical_error(tmp_path: Path) -> None:
    circuit_dir = (
        tmp_path
        / "code=surface_code_Z,d=5,rounds=5,noisemodel=si1000,p=0.001,xyz=False"
    )
    decoding_dir = circuit_dir / "decoding_result"

    forced_dir = decoding_dir / "decoder=BP-LSD,metric=forced_gap_ml"
    linearize_dir = decoding_dir / "decoder=BP-LSD,metric=linearize_logicalgap"

    _write_batch(forced_dir, "forced_gap_ml", [0.2, 0.8], [False, True])
    _write_batch(linearize_dir, "linearize_logicalgap", [1.0, 2.0], [True, False])

    manager = SimulationDataManager(tmp_path)
    config = PlotConfig(
        metric_name="forced_gap_ml",
        decoder_names=["BP-LSD"],
        filters={"d": 5, "p": 0.001},
        plot_params={"use_linearize": True},
    )

    df = (
        manager.query(config)
        .select(["shot_id", "forced_gap_ml", "is_logical_error"])
        .collect()
        .sort("shot_id")
    )

    assert df["forced_gap_ml"].to_list() == [0.2, 0.8]
    assert df["is_logical_error"].to_list() == [True, False]


def test_postselect_spec_passes_use_linearize_to_plot_config() -> None:
    captured: dict[str, object] = {}

    class DummyManager:
        def query(self, config: PlotConfig) -> pl.LazyFrame:
            captured["use_linearize"] = config.extra_options.get("use_linearize")
            return pl.DataFrame(
                {
                    "forced_gap_ml": [0.2, 0.8],
                    "is_logical_error": [True, False],
                }
            ).lazy()

    fig, ax = plt.subplots()
    spec = PostSelectSpec(
        metric_name="forced_gap_ml",
        decoder_names=["BP-LSD"],
        use_linearize=True,
    )

    PostSelectionPlotter(DummyManager()).plot([spec], ax, num_points=2)

    assert captured["use_linearize"] is True
    plt.close(fig)


def test_postselect_mark_zero_gap_skips_star_when_no_logical_errors_remain() -> None:
    class DummyManager:
        def query(self, config: PlotConfig) -> pl.LazyFrame:
            return pl.DataFrame(
                {
                    "linearize_logicalgap": [-1.0, 0.0, 1.0, 2.0],
                    "is_logical_error": [True, False, False, False],
                }
            ).lazy()

    fig, ax = plt.subplots()
    spec = PostSelectSpec(
        metric_name="linearize_logicalgap",
        mark_zero_gap=True,
    )

    PostSelectionPlotter(DummyManager()).plot([spec], ax, num_points=3)

    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert np.isnan(line.get_ydata()[1:]).all()
    plt.close(fig)


def test_postselect_mark_zero_gap_adds_star_without_legend_entry() -> None:
    class DummyManager:
        def query(self, config: PlotConfig) -> pl.LazyFrame:
            return pl.DataFrame(
                {
                    "linearize_logicalgap": [-1.0, 0.0, 1.0, 2.0],
                    "is_logical_error": [False, False, True, False],
                }
            ).lazy()

    fig, ax = plt.subplots()
    spec = PostSelectSpec(
        metric_name="linearize_logicalgap",
        mark_zero_gap=True,
    )

    PostSelectionPlotter(DummyManager()).plot([spec], ax, num_points=3)

    assert len(ax.lines) == 2
    star_line = ax.lines[1]
    assert star_line.get_marker() == "*"
    assert star_line.get_label() == "_nolegend_"
    assert list(star_line.get_xdata()) == [0.5]
    assert list(star_line.get_ydata()) == [0.5]
    plt.close(fig)
