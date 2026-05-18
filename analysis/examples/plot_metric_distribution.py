"""Example: plot metric distributions from QEC simulation results.

Run from the repository root::

    python -m analysis.examples.plot_metric_distribution

The script demonstrates three typical use-cases:
    1. Numeric metric (logical_gap) with no grouping.
    2. Numeric metric grouped by decoder, split by is_logical_error.
    3. Boolean metric (ar-lec) accept-rate bar chart grouped by decoder.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis.src import (
    BooleanMetricAnalyzer,
    NumericMetricAnalyzer,
    PlotConfig,
    SimulationDataManager,
)

# ---------------------------------------------------------------------------
# Point this at your actual result directory.
# The example uses the bundled example_outdir from the repository root.
# ---------------------------------------------------------------------------
RESULT_DIR = Path(__file__).parents[2] / "example_outdir"


def example_numeric_no_grouping(manager: SimulationDataManager) -> None:
    """Single histogram of logical_gap over all available shots."""
    config = PlotConfig(
        metric_name="logical_gap",
        filters={"d": 5, "p": 0.001},
        plot_params={"bins": 40, "alpha": 0.75},
    )
    lf = manager.query(config)

    fig, ax = plt.subplots()
    ax.set_title("logical_gap distribution (all shots)")
    NumericMetricAnalyzer().plot_distribution(lf, config, ax)
    fig.tight_layout()
    fig.savefig("logical_gap_all.png", dpi=150)
    print("Saved logical_gap_all.png")
    plt.close(fig)


def example_numeric_grouped_by_error(manager: SimulationDataManager) -> None:
    """Logical_gap histograms split by whether a logical error occurred."""
    config = PlotConfig(
        metric_name="logical_gap",
        filters={"d": 5, "p": 0.001},
        # No group_by columns beyond the logical-error split
        separate_logical_error=True,
        plot_params={"bins": 40, "alpha": 0.65},
    )
    lf = manager.query(config)

    fig, ax = plt.subplots()
    ax.set_title("logical_gap: success vs. logical-error shots")
    NumericMetricAnalyzer().plot_distribution(lf, config, ax)
    fig.tight_layout()
    fig.savefig("logical_gap_by_error.png", dpi=150)
    print("Saved logical_gap_by_error.png")
    plt.close(fig)


def example_numeric_grouped_by_decoder(manager: SimulationDataManager) -> None:
    """Overlay logical_gap histograms for each decoder side-by-side."""
    config = PlotConfig(
        metric_name="logical_gap",
        filters={"d": 5, "p": 0.001},
        group_by=["decoder"],
        separate_logical_error=True,
        plot_params={"bins": 40, "alpha": 0.65},
    )
    lf = manager.query(config)

    fig, ax = plt.subplots()
    ax.set_title("logical_gap by decoder (split by logical error)")
    NumericMetricAnalyzer().plot_distribution(lf, config, ax)
    fig.tight_layout()
    fig.savefig("logical_gap_by_decoder.png", dpi=150)
    print("Saved logical_gap_by_decoder.png")
    plt.close(fig)


def example_boolean_accept_rate(manager: SimulationDataManager) -> None:
    """Accept-rate bar chart for the ar-lec metric, grouped by decoder."""
    config = PlotConfig(
        metric_name="ar-lec",
        filters={"d": 5, "p": 0.001},
        group_by=["decoder"],
        separate_logical_error=True,
        plot_params={"alpha": 0.8},
    )
    lf = manager.query(config)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("ar-lec accept rate by decoder (split by logical error)")
    BooleanMetricAnalyzer().plot_distribution(lf, config, ax)
    fig.tight_layout()
    fig.savefig("ar_lec_accept_rate.png", dpi=150)
    print("Saved ar_lec_accept_rate.png")
    plt.close(fig)


def example_batch_subset(manager: SimulationDataManager) -> None:
    """Load only batch index 5 to limit memory usage."""
    config = PlotConfig(
        metric_name="logical_gap",
        filters={"d": 5, "p": 0.001},
        batch_indices=[5],          # only load batch 5
        plot_params={"bins": 40, "alpha": 0.75},
    )
    lf = manager.query(config)

    fig, ax = plt.subplots()
    ax.set_title("logical_gap — batch 5 only")
    NumericMetricAnalyzer().plot_distribution(lf, config, ax)
    fig.tight_layout()
    fig.savefig("logical_gap_batch5.png", dpi=150)
    print("Saved logical_gap_batch5.png")
    plt.close(fig)


if __name__ == "__main__":
    manager = SimulationDataManager(RESULT_DIR)

    # Run whichever examples are relevant for the available data.
    # Wrap each in try/except so a missing metric doesn't abort the rest.
    for fn in [
        example_numeric_no_grouping,
        example_numeric_grouped_by_error,
        example_numeric_grouped_by_decoder,
        example_batch_subset,
    ]:
        try:
            fn(manager)
        except FileNotFoundError as exc:
            print(f"[skip] {fn.__name__}: {exc}")

    # ar-lec is only available when the BP-LSD decoder has been run
    try:
        example_boolean_accept_rate(manager)
    except FileNotFoundError as exc:
        print(f"[skip] example_boolean_accept_rate: {exc}")
