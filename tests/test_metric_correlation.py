from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import pytest

from analysis.src.metric_correlation import (
    MetricPairConfig,
    _confidence_probability_table,
    plot_metric_scatter,
)


def test_confidence_probability_table() -> None:
    df = pl.DataFrame(
        {
            "logical_gap": [1.0, 1.0, 2.0, 2.0],
            "forced_gap_ml": [2.0, 0.5, 2.0, 3.0],
            "is_logical_error_x": [True, True, False, False],
        }
    )

    table = _confidence_probability_table(df)

    assert table["subset"].to_list() == [
        "overall",
        "logical_error",
        "non_logical_error",
    ]
    assert table["n"].to_list() == [4, 2, 2]
    assert table["p_overconfident"].to_list() == [0.5, 0.5, 0.5]
    assert table["p_underconfident"].to_list() == [0.25, 0.5, 0.0]
    assert table["p_equal"].to_list() == [0.25, 0.0, 0.5]


def test_plot_metric_scatter_rejects_other_pairs() -> None:
    config = MetricPairConfig(
        x_metric="logical_gap",
        y_metric="cluster_llr",
        report_confidence_statistics=True,
    )

    with pytest.raises(ValueError, match="report_confidence_statistics"):
        plot_metric_scatter(object(), config)


def test_plot_metric_scatter_prints_confidence_stats(monkeypatch, capsys) -> None:
    df = pl.DataFrame(
        {
            "logical_gap": [1.0, 1.0, 2.0, 2.0],
            "forced_gap_ml": [2.0, 0.5, 2.0, 3.0],
            "is_logical_error_x": [True, True, False, False],
        }
    )
    monkeypatch.setattr(
        "analysis.src.metric_correlation._load_and_join",
        lambda manager, config: df,
    )

    fig, ax = plt.subplots()
    config = MetricPairConfig(
        x_metric="logical_gap",
        y_metric="forced_gap_ml",
        report_confidence_statistics=True,
    )

    plot_metric_scatter(object(), config, ax=ax, print_ccc=False)
    captured = capsys.readouterr().out

    assert "Confidence probabilities for logical_gap vs forced_gap_ml" in captured
    assert "logical_error" in captured

    plt.close(fig)