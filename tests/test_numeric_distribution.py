from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

from analysis.src.analyzers import NumericMetricAnalyzer
from analysis.src.config import PlotConfig


def test_numeric_distribution_separates_logical_error_with_fixed_styles() -> None:
    lf = pl.DataFrame(
        {
            "linearize_logicalgap": [1.0, 1.0, 2.0, 2.0],
            "is_logical_error": [False, False, True, True],
        }
    ).lazy()
    config = PlotConfig(
        metric_name="linearize_logicalgap",
        separate_logical_error=True,
    )

    fig, ax = plt.subplots()
    NumericMetricAnalyzer().plot_distribution(lf, config, ax)

    assert len(ax.collections) == 2

    non_error, logical_error = ax.collections
    assert non_error.get_edgecolors()[0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert logical_error.get_edgecolors()[0].tolist() == [1.0, 0.0, 0.0, 1.0]

    non_error_vertices = non_error.get_paths()[0].vertices
    logical_error_vertices = logical_error.get_paths()[0].vertices
    assert len(non_error_vertices) > 4
    assert len(logical_error_vertices) == 4

    plt.close(fig)
