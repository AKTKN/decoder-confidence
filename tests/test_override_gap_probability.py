from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import polars as pl

from analysis.src.case_histogram import (
    LogicalGapSplitConfig,
    OverrideProbabilityAnalyzer,
)


def test_override_gap_histogram_plots_probability_for_both_gap_metrics(monkeypatch) -> None:
    df = pl.DataFrame(
        {
            "linearize_logicalgap": [-1.0, -1.0, 1.0, 1.0],
            "logical_gap": [0.0, 0.0, 1.0, 1.0],
            "forced_gap_ml": [0.0, 1.0, 1.0, 2.0],
            "forced_gap_ml_case": [3, 3, -1, -1],
            "is_logical_error_fg": [False, False, False, False],
        }
    )
    monkeypatch.setattr(
        "analysis.src.case_histogram._load_logical_gap_split_data",
        lambda manager, config: df,
    )

    fig, ax = plt.subplots()
    config = LogicalGapSplitConfig(round_digits=0)
    OverrideProbabilityAnalyzer().plot_override_gap_histogram(object(), config, ax)

    scatter_collections = [
        coll for coll in ax.collections if isinstance(coll, PathCollection)
    ]
    assert len(scatter_collections) == 2

    exact_offsets = scatter_collections[0].get_offsets()
    forced_offsets = scatter_collections[1].get_offsets()
    exact_points = sorted((round(float(x), 6), round(float(y), 6)) for x, y in exact_offsets)
    forced_points = sorted((round(float(x), 6), round(float(y), 6)) for x, y in forced_offsets)

    assert exact_points == [(0.0, 1.0), (1.0, 0.0)]
    assert forced_points == [(0.0, 1.0), (1.0, 0.5), (2.0, 0.0)]

    handles, labels = ax.get_legend_handles_labels()
    assert len(handles) == 2
    assert labels == ["Exact gap", "Forced gap"]

    plt.close(fig)


def test_override_gap_histogram_excludes_zero_when_include_0_is_false(monkeypatch) -> None:
    df = pl.DataFrame(
        {
            "linearize_logicalgap": [0.0, -1.0, 1.0, 0.0],
            "logical_gap": [0.0, 0.0, 1.0, 1.0],
            "forced_gap_ml": [0.0, 1.0, 1.0, 2.0],
            "forced_gap_ml_case": [3, 3, -1, -1],
            "is_logical_error_fg": [False, False, False, False],
        }
    )
    monkeypatch.setattr(
        "analysis.src.case_histogram._load_logical_gap_split_data",
        lambda manager, config: df,
    )

    fig, ax = plt.subplots()
    config = LogicalGapSplitConfig(round_digits=0, include_0=False)
    OverrideProbabilityAnalyzer().plot_override_gap_histogram(object(), config, ax)

    scatter_collections = [
        coll for coll in ax.collections if isinstance(coll, PathCollection)
    ]
    assert len(scatter_collections) == 2

    exact_offsets = scatter_collections[0].get_offsets()
    forced_offsets = scatter_collections[1].get_offsets()
    exact_points = sorted((round(float(x), 6), round(float(y), 6)) for x, y in exact_offsets)
    forced_points = sorted((round(float(x), 6), round(float(y), 6)) for x, y in forced_offsets)

    assert exact_points == [(0.0, 0.5), (1.0, 0.0)]
    assert forced_points == [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)]

    plt.close(fig)
