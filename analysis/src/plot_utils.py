"""Notebook-facing helpers: data plotting calls + figure finalization.

Each ``draw_*`` function does exactly two things: query data via
:class:`~analysis.src.data_manager.SimulationDataManager` and call the
matching analyzer/plotter method on the given axes. They never set
title/labels/legend/grid/scale — that is the notebook's responsibility,
applied directly to ``ax`` between the ``draw_*`` call and
:func:`finalize_plot`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from analysis.src.analyzers import (
    BooleanMetricAnalyzer,
    ConditionalLERAnalyzer,
    NumericMetricAnalyzer,
)
from analysis.src.case_histogram import (
    CaseScatterConfig,
    ForcedGapMLCaseAnalyzer,
    ForcedGapMLCaseConfig,
)
from analysis.src.config import ConditionalLERConfig, PlotConfig
from analysis.src.data_manager import SimulationDataManager
from analysis.src.figure_style import finalize_axes
from analysis.src.postselect import PostSelectionPlotter, PostSelectSpec
from analysis.src.relative_improvement import RelativeImprovementPlotter, RelImprovSpec


# ---------------------------------------------------------------------------
# Output / finalization
# ---------------------------------------------------------------------------

@dataclass
class PlotOutputOptions:
    """Save/show/close behaviour for :func:`finalize_plot`."""

    save_path: Path | None = None
    dpi: int = 300
    show: bool = True
    close: bool = True


def finalize_plot(
    fig: plt.Figure,
    ax: plt.Axes | Sequence[plt.Axes],
    output: PlotOutputOptions | None = None,
) -> None:
    """Apply :func:`~analysis.src.figure_style.finalize_axes` and save/show/close.

    Parameters
    ----------
    fig:
        The figure to finalize.
    ax:
        A single ``Axes`` or any iterable of ``Axes`` (e.g. from
        ``plt.subplots(2, 2)``).
    output:
        Save/show/close options. Defaults to showing and closing the figure
        without saving.
    """
    output = output or PlotOutputOptions()

    axes: Iterable[plt.Axes]
    if isinstance(ax, plt.Axes):
        axes = [ax]
    else:
        axes = np.asarray(ax).flatten()

    for a in axes:
        finalize_axes(a)

    if output.save_path is not None:
        fig.savefig(output.save_path, dpi=output.dpi)
        print(f"Saved {output.save_path}")
    if output.show:
        plt.show()
    if output.close:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Draw helpers (data only)
# ---------------------------------------------------------------------------

def draw_numeric_distribution(
    manager: SimulationDataManager,
    config: PlotConfig,
    ax: plt.Axes,
) -> plt.Axes:
    """Query and draw a numeric metric distribution onto *ax*."""
    lf = manager.query(config)
    NumericMetricAnalyzer().plot_distribution(lf, config, ax)
    return ax


def draw_boolean_distribution(
    manager: SimulationDataManager,
    config: PlotConfig,
    ax: plt.Axes,
) -> plt.Axes:
    """Query and draw a boolean metric's accept-rate distribution onto *ax*."""
    lf = manager.query(config)
    BooleanMetricAnalyzer().plot_distribution(lf, config, ax)
    return ax


def draw_conditional_ler(
    manager: SimulationDataManager,
    config: ConditionalLERConfig,
    ax: plt.Axes,
) -> plt.Axes:
    """Query and draw P(logical error | metric) onto *ax*."""
    lf = manager.query(config)
    ConditionalLERAnalyzer().plot_conditional_ler(lf, config, ax)
    return ax


def draw_conditional_ler_fitting(
    manager: SimulationDataManager,
    config: ConditionalLERConfig,
    ax: plt.Axes,
) -> plt.Axes:
    """Query and draw the log-odds scatter + linear fit onto *ax*."""
    lf = manager.query(config)
    ConditionalLERAnalyzer().plot_fitting(lf, config, ax)
    return ax


def draw_post_selection(
    manager: SimulationDataManager,
    specs: Sequence[PostSelectSpec],
    ax: plt.Axes,
    **plot_kw: Any,
) -> plt.Axes:
    """Draw post-selection (abort rate vs. LER) curves onto *ax*."""
    PostSelectionPlotter(manager).plot(list(specs), ax, **plot_kw)
    return ax


def draw_relative_improvement(
    manager: SimulationDataManager,
    base_spec: RelImprovSpec,
    comparison_specs: Sequence[RelImprovSpec],
    ax: plt.Axes,
    **plot_kw: Any,
) -> plt.Axes:
    """Draw relative-improvement curves onto *ax*."""
    RelativeImprovementPlotter(manager).plot(
        base_spec=base_spec,
        comparison_specs=list(comparison_specs),
        ax=ax,
        **plot_kw,
    )
    return ax


def draw_case_histogram(
    manager: SimulationDataManager,
    config: ForcedGapMLCaseConfig,
    ax: plt.Axes,
) -> plt.Axes:
    """Draw the forced_gap_ml_case label histogram onto *ax*."""
    ForcedGapMLCaseAnalyzer().plot_case_histogram(manager, config, ax)
    return ax


def draw_case_scatter(
    manager: SimulationDataManager,
    config: CaseScatterConfig,
    ax: plt.Axes,
    *,
    scatter_kw: dict[str, Any] | None = None,
) -> plt.Axes:
    """Draw a case-filtered forced_gap_ml vs. other-metric scatter onto *ax*."""
    return ForcedGapMLCaseAnalyzer().plot_case_scatter(manager, config, ax, scatter_kw=scatter_kw)
