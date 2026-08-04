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
from analysis.src.anchored_reselection import (
    AnchoredReselectionAnalyzer,
    AnchoredReselectionConfig,
    BinScale,
    TransitionCategory,
    DEFAULT_CATEGORY_ORDER,
)
from analysis.src.case_histogram import (
    CaseScatterConfig,
    ForcedGapMLCaseAnalyzer,
    ForcedGapMLCaseConfig,
    LogicalGapSplitAnalyzer,
    LogicalGapSplitConfig,
    OverrideProbabilityAnalyzer,
    OverrideProbabilityConfig,
)
from analysis.src.config import ConditionalLERConfig, PlotConfig
from analysis.src.data_manager import SimulationDataManager
from analysis.src.figure_style import finalize_axes
from analysis.src.postselect import PostSelectionPlotter, PostSelectSpec
from analysis.src.relative_improvement import RelativeImprovementPlotter, RelImprovSpec
from analysis.src.ranking_degradation import (
    RankingAnalysisConfig,
    RankingDegradationAnalyzer,
    RankingMetricName,
    ScoreName,
    ThresholdRejectionSet,
)
from analysis.src.stage_gap_difference import (
    GapDifferenceStat,
    OverrideThresholdGap,
    Stage1GapDifferenceAnalyzer,
    Stage1GapDifferenceConfig,
    Stage1WeightCase,
)


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
    ax: plt.Axes | tuple[plt.Axes, plt.Axes],
) -> plt.Axes | tuple[plt.Axes, plt.Axes]:
    """Query and draw P(logical error | metric) onto *ax*.

    When ``config.split_by_sign`` is ``True``, *ax* must be a 2-tuple
    ``(ax_negative, ax_positive)``: shots with ``metric <= 0`` are drawn on
    ``ax_negative`` and shots with ``metric > 0`` on ``ax_positive``, each
    with its own fit.
    """
    lf = manager.query(config)
    ConditionalLERAnalyzer().plot_conditional_ler(lf, config, ax)
    return ax


def draw_conditional_ler_fitting(
    manager: SimulationDataManager,
    config: ConditionalLERConfig,
    ax: plt.Axes | tuple[plt.Axes, plt.Axes],
) -> plt.Axes | tuple[plt.Axes, plt.Axes]:
    """Query and draw the log-odds scatter + linear fit onto *ax*.

    When ``config.split_by_sign`` is ``True``, *ax* must be a 2-tuple
    ``(ax_negative, ax_positive)`` (see :func:`draw_conditional_ler`).
    """
    lf = manager.query(config)
    ConditionalLERAnalyzer().plot_fitting(lf, config, ax)
    return ax


def draw_post_selection(
    manager: SimulationDataManager,
    specs: Sequence[PostSelectSpec],
    ax: plt.Axes,
    **plot_kw: Any,
) -> plt.Axes:
    """Draw post-selection (abort rate vs. LER) curves onto *ax*.

    Pass ``report_abort_rates=[0.1, 0.3, ...]`` to also print, for each
    spec, the post-selected LER at the curve point closest to each of
    those abort rates (see :meth:`PostSelectionPlotter.plot`).
    """
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


def draw_logical_gap_split_by_case(
    manager: SimulationDataManager,
    config: LogicalGapSplitConfig,
    ax: plt.Axes,
    *,
    case_values: list[int] | None = None,
    bar_kw: dict[str, Any] | None = None,
) -> plt.Axes:
    """Draw five normalized logical_gap histograms split by linearize_logicalgap sign and case."""
    return LogicalGapSplitAnalyzer().plot_split_by_sign_and_case(
        manager, config, ax, case_values=case_values, bar_kw=bar_kw,
    )


def draw_override_probability(
    manager: SimulationDataManager,
    config: OverrideProbabilityConfig,
    ax: plt.Axes,
    *,
    style: str = "bar",
    **plot_kw: Any,
) -> plt.Axes:
    """Draw P(linearize_logicalgap <= threshold | logical_gap) onto *ax*.

    ``style="bar"`` (default) draws a bar chart with Wilson 95% CI error
    bars -- suitable for noise models where ``logical_gap`` takes few
    discrete values (e.g. phenomenological noise).
    ``style="scatter"`` draws a scatter plot with a shaded Wilson CI band --
    suitable for noise models where ``logical_gap`` is effectively
    continuous (e.g. circuit-level noise).
    """
    analyzer = OverrideProbabilityAnalyzer()
    if style == "bar":
        return analyzer.plot_bar(manager, config, ax, **plot_kw)
    if style == "scatter":
        return analyzer.plot_scatter(manager, config, ax, **plot_kw)
    raise ValueError(f"Unknown style {style!r}; expected 'bar' or 'scatter'")


def draw_override_gap_histogram(
    manager: SimulationDataManager,
    config: LogicalGapSplitConfig,
    ax: plt.Axes,
    *,
    bar_kw: dict[str, Any] | None = None,
) -> plt.Axes:
    """Draw scatter + error-bar comparisons of ``P(override | gap)`` for two gap metrics.

    The two series are computed independently from the same shot table:

    - exact gap: ``P(linearize_logicalgap <= threshold | logical_gap)``
    - forced gap: ``P(linearize_logicalgap <= threshold | forced_gap_ml)``
    """
    return OverrideProbabilityAnalyzer().plot_override_gap_histogram(
        manager, config, ax, bar_kw=bar_kw,
    )


def draw_stage1_gap_difference_by_logical_gap(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
    ax: plt.Axes,
    *,
    stat_name: GapDifferenceStat = "MAE",
    bar_kw: dict[str, Any] | None = None,
) -> plt.Axes:
    """Draw a bar chart of Δ_fg − Δ_ex statistics conditioned on logical_gap."""
    return Stage1GapDifferenceAnalyzer().plot_stat_by_logical_gap(
        manager,
        config,
        ax,
        stat_name=stat_name,
        bar_kw=bar_kw,
    )


def draw_stage1_logical_gap_distribution(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
    ax: plt.Axes,
    *,
    stage1_weight_cases: Sequence[Stage1WeightCase] = ("all",),
    normalize: bool = False,
    bar_kw: dict[str, Any] | None = None,
) -> plt.Axes:
    """Draw logical_gap distributions split by stage-1 weight case."""
    return Stage1GapDifferenceAnalyzer().plot_logical_gap_distribution(
        manager,
        config,
        ax,
        stage1_weight_cases=stage1_weight_cases,
        normalize=normalize,
        bar_kw=bar_kw,
    )


def draw_stage1_forced_gap_conditional_heatmap(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
    ax: plt.Axes,
    *,
    cmap: str = "viridis",
    add_colorbar: bool = True,
    colorbar_label: str = r"$P(\Delta_{\rm fg}\mid\Delta_{\rm ex})$",
    signed_by_logical_error: bool = False,
) -> plt.Axes:
    """Draw a log-normalized heatmap of P(Δ_fg | Δ_ex)."""
    return Stage1GapDifferenceAnalyzer().plot_forced_gap_conditional_heatmap(
        manager,
        config,
        ax,
        cmap=cmap,
        add_colorbar=add_colorbar,
        colorbar_label=colorbar_label,
        signed_by_logical_error=signed_by_logical_error,
    )


def draw_override_threshold_logical_error_curve(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
    ax: plt.Axes,
    *,
    gap_name: OverrideThresholdGap = "forced_gap_ml",
    thresholds: Sequence[float] | None = None,
    plot_kw: dict[str, Any] | None = None,
) -> plt.Axes:
    """Draw conditional logical-error rate vs gap threshold inside override shots."""
    return Stage1GapDifferenceAnalyzer().plot_override_threshold_logical_error_curve(
        manager,
        config,
        ax,
        gap_name=gap_name,
        thresholds=thresholds,
        plot_kw=plot_kw,
    )


def draw_ranking_degradation(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    ax: plt.Axes,
    *,
    metrics: Sequence[RankingMetricName] = (
        "forced_only_fraction",
        "jaccard_distance",
        "symmetric_difference_fraction",
    ),
    exact_score: ScoreName = "exact_gap",
    comparison_score: ScoreName = "forced_gap",
) -> plt.Axes:
    """Draw abort-rate based accepted-set degradation metrics."""
    return RankingDegradationAnalyzer().plot_ranking_degradation(
        manager,
        config,
        ax,
        metrics=metrics,
        exact_score=exact_score,
        comparison_score=comparison_score,
    )


def draw_ranking_postselection_curves(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    ax: plt.Axes,
    *,
    error_source: str = "score_default",
    shade_ci: bool = True,
) -> plt.Axes:
    """Draw post-selected LER curves for ranking scores."""
    return RankingDegradationAnalyzer().plot_postselection_curves(
        manager,
        config,
        ax,
        error_source=error_source,  # type: ignore[arg-type]
        shade_ci=shade_ci,
    )


def draw_false_safe_case_fractions(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    ax: plt.Axes,
    *,
    abort_rates: Sequence[float],
    exact_score: ScoreName = "exact_gap",
    comparison_score: ScoreName = "forced_gap",
) -> plt.Axes:
    """Draw indicator fractions inside shots accepted only by the comparison score."""
    return RankingDegradationAnalyzer().plot_false_safe_case_fractions(
        manager,
        config,
        ax,
        abort_rates=abort_rates,
        exact_score=exact_score,
        comparison_score=comparison_score,
    )


def draw_gap_threshold_rejection_fractions(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    ax: plt.Axes,
    *,
    thresholds: Sequence[float] | None = None,
    forced_logical_error: bool | None = None,
    sets: Sequence[ThresholdRejectionSet] = (
        "both_rejected",
        "overconfident",
        "underconfident",
    ),
) -> plt.Axes:
    """Draw threshold-based rejection-set fractions normalized by forced rejects."""
    return RankingDegradationAnalyzer().plot_gap_threshold_rejection_fractions(
        manager,
        config,
        ax,
        thresholds=thresholds,
        forced_logical_error=forced_logical_error,
        sets=sets,
    )


def draw_gap_threshold_case_fractions(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    ax: plt.Axes,
    *,
    threshold: float,
) -> plt.Axes:
    """Draw stage/override category fractions for one logical-gap threshold."""
    return RankingDegradationAnalyzer().plot_gap_threshold_case_fractions(
        manager,
        config,
        ax,
        threshold=threshold,
    )


def draw_reselection_transition(
    manager: SimulationDataManager,
    config: AnchoredReselectionConfig,
    ax: plt.Axes,
    *,
    bins: int | Sequence[float] = 10,
    bin_scale: BinScale = "gap_value",
    x_range: tuple[float, float] | None = None,
    category_order: Sequence[TransitionCategory] = DEFAULT_CATEGORY_ORDER,
    bar_kw: dict[str, Any] | None = None,
) -> plt.Axes:
    """Draw the stacked-bar reselection-transition chart (diagnostic plot a).

    Shows, within the negative-anchored subset ``A = {Delta_anc < 0}``, the
    conditional fractions of shots that are fixed / broken / left wrong /
    left correct by the ordinary forced-gap reselection.  ``bin_scale``
    selects whether bars are binned by raw ``Delta_fg`` value (default) or
    by the corresponding forced-gap abort rate, which puts the x-axis on the
    same rejection-rate scale as :func:`draw_postselection_decomposition`.
    ``x_range`` optionally fixes the ``(min, max)`` binned range (units of
    ``bin_scale``) instead of using the observed range within ``A``; values
    outside it are clamped into the first/last bin.
    """
    return AnchoredReselectionAnalyzer().plot_reselection_transition(
        manager,
        config,
        ax,
        bins=bins,
        bin_scale=bin_scale,
        x_range=x_range,
        category_order=category_order,
        bar_kw=bar_kw,
    )


def draw_postselection_decomposition(
    manager: SimulationDataManager,
    config: AnchoredReselectionConfig,
    ax: plt.Axes,
    *,
    abort_rates: Sequence[float] | None = None,
    num_points: int = 30,
    max_multiple: float = 2.0,
    alpha: float = 0.05,
    mark_r_minus: bool = True,
    relative: bool = False,
    plot_kw: dict[str, Any] | None = None,
) -> plt.Axes:
    """Draw the decision/ranking decomposition of the post-selection advantage (plot b).

    ``relative=False`` (default) plots ``D_decision(r)``, ``D_ranking(r)``,
    and their sum ``D_total(r) = L_fg(r) - L_anc(r)`` as functions of the
    matched rejection rate ``r``, with a vertical marker at ``r_minus =
    P(Delta_anc < 0)``.

    ``relative=True`` plots only ``D_total_relative(r) = L_fg(r) / L_anc(r)``
    instead -- a single ratio curve, better suited to a log y-axis than the
    absolute difference (which shrinks toward 0 even when the two LERs
    differ by a large multiplicative factor).
    """
    return AnchoredReselectionAnalyzer().plot_postselection_decomposition(
        manager,
        config,
        ax,
        abort_rates=abort_rates,
        num_points=num_points,
        max_multiple=max_multiple,
        alpha=alpha,
        mark_r_minus=mark_r_minus,
        relative=relative,
        plot_kw=plot_kw,
    )
