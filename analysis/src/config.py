from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Metrics supported by :class:`ConditionalLERAnalyzer`.
CONDITIONAL_LER_SUPPORTED_METRICS: frozenset[str] = frozenset(
    {"logical_gap", "linearize_logicalgap", "forced_gap_ml"}
)


def normalize_metric_names(metric_name: str | List[str]) -> List[str]:
    """Return a normalized list of metric names.

    ``metric_name`` can be a single string or a list of strings. The returned
    list preserves the given order.
    """
    if isinstance(metric_name, str):
        return [metric_name]
    if isinstance(metric_name, list):
        if not metric_name:
            raise ValueError("metric_name list must be non-empty")
        if not all(isinstance(name, str) for name in metric_name):
            raise ValueError("metric_name list must contain only strings")
        return metric_name
    raise ValueError("metric_name must be a string or list of strings")


@dataclass
class PlotConfig:
    """Encapsulates all configuration for a metric distribution plot.

    Parameters
    ----------
    metric_name:
        Name of the metric column to analyse (e.g. ``"logical_gap"``, ``"ar-lec"``)
        or a list of metric names to overlay on the same axes.
    decoder_names:
        Subset of decoder names to include. ``None`` includes every decoder found on
        disk for the requested metric.
    filters:
        Key/value conditions applied at directory-scan time to restrict which
        circuit-parameter directories are loaded (e.g. ``{"d": 5, "p": 0.001}``).
        Integer, float and boolean values are coerced for comparison.
    group_by:
        Column names whose combinations produce separate series on the same axes
        (e.g. ``["p", "decoder"]``). Values must be present as columns in the
        LazyFrame returned by :class:`SimulationDataManager`.
    separate_logical_error:
        When ``True``, ``"is_logical_error"`` is appended to the effective partition
        keys so that shots with and without logical errors are plotted separately.
    plot_params:
        Keyword arguments forwarded verbatim to the underlying matplotlib primitive
        (e.g. ``{"bins": 50, "alpha": 0.7}``).  Special keys consumed internally:
        ``"bins"``, ``"alpha_ci"``, ``"shade_alpha"``, ``"round_digits"``,
        ``"use_negative_gap"``, ``"convert_db"``, ``"use_linearize"``.
    extra_options:
        Reserved for future extensibility (e.g. conditional LER, post-selection).
    batch_indices:
        Restrict loading to specific batch indices (1-based integers matching the
        ``batch=N`` suffix in parquet filenames). ``None`` loads all available batches.
    metric_labels:
        Mapping from internal metric name to a human-readable display label used in
        legend entries (e.g. ``{"linearize_logicalgap": "BP+OSD"}``).
        Metric names absent from the mapping fall back to the raw name.
    """

    metric_name: str | List[str]
    decoder_names: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    separate_logical_error: bool = False
    plot_params: Dict[str, Any] = field(default_factory=dict)
    extra_options: Dict[str, Any] = field(default_factory=dict)
    batch_indices: Optional[List[int]] = None
    metric_labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConditionalLERConfig:
    """Configuration for :class:`~analysis.src.analyzers.ConditionalLERAnalyzer`.

    Parameters
    ----------
    metric_name:
        Continuous metric used as the x-axis (e.g. ``"logical_gap"``,
        ``"linearize_logicalgap"``, ``"forced_gap_ml"``). May be a list to overlay
        multiple metrics. Each metric must be one of
        :data:`CONDITIONAL_LER_SUPPORTED_METRICS`.
    decoder_names:
        Subset of decoder names to include. ``None`` includes every decoder.
    filters:
        Circuit-parameter filters applied at directory-scan time (same
        semantics as :attr:`PlotConfig.filters`).
    group_by:
        Column names whose unique combinations produce separate series.
    bins:
        Number of uniform bins along the metric axis. ``None`` disables
        binning and uses each unique metric value directly.
    alpha:
        Significance level for Wilson CI (default 0.05 → 95 % CI).
    get_fitting_plot:
        When ``True``, :meth:`~ConditionalLERAnalyzer.plot_fitting` will
        draw a log-odds scatter with a linear fit line.
    show_sigmoid_fit:
        When ``True``, :meth:`~ConditionalLERAnalyzer.plot_conditional_ler`
        additionally draws a fitted curve ``y(g) = 1 / (1 + exp(k * g))``,
        where ``k`` is obtained from an origin-constrained (``l = 0``) least
        squares fit of ``k * g = log((1 - y) / y)`` to the binned conditional
        LER values.
    batch_indices:
        Restrict loading to specific batch indices. ``None`` loads all batches.
    round_digits:
        Decimal digits to round metric values before analysis. ``None`` leaves
        values unchanged.
    convert_db:
        When ``True``, convert gap metric values to decibels before binning,
        rounding, and plotting: ``gap_dB = (10 / ln(10)) * gap``.
        Only valid for ``"logical_gap"``, ``"linearize_logicalgap"``, and
        ``"forced_gap_ml"``.
    metric_labels:
        Mapping from internal metric name to a human-readable display label used in
        legend entries (e.g. ``{"logical_gap": "BP decoder"}``).
        Metric names absent from the mapping fall back to the raw name.
    split_by_sign:
        When ``True``, split the data into a non-positive group
        (``metric <= 0``) and a positive group (``metric > 0``) and plot each
        group on its own axes. Fitting (``show_sigmoid_fit`` and
        ``get_fitting_plot``) is performed independently for each group.
        When enabled, ``ax`` passed to
        :meth:`~ConditionalLERAnalyzer.plot_conditional_ler` and
        :meth:`~ConditionalLERAnalyzer.plot_fitting` must be a 2-element
        sequence ``(ax_negative, ax_positive)``. Mainly intended for metrics
        that take both signs, e.g. ``"linearize_logicalgap"``.
    extra_options:
        Reserved for future extensibility (e.g. decoder filters). Consumed by
        :class:`~analysis.src.data_manager.SimulationDataManager`, which
        shares its directory-scanning logic with :class:`PlotConfig`.
    write_forced_unconv_line:
        Only supported for ``"forced_gap_ml"`` and ``"linearize_logicalgap"``.
        When ``True``, :meth:`~ConditionalLERAnalyzer.plot_conditional_ler`
        additionally draws a horizontal dotted line at the conditional
        logical-error rate among shots whose stage-2 forced decode never
        converged (metric value ``+inf``, off the finite-gap x-range) --
        i.e. ``P(logical_error | metric == +inf)`` -- in the same color as
        that metric's/group's curve.
    """

    metric_name: str | List[str]
    decoder_names: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    bins: Optional[int] = None
    alpha: float = 0.05
    get_fitting_plot: bool = False
    show_sigmoid_fit: bool = False
    batch_indices: Optional[List[int]] = None
    round_digits: Optional[int] = None
    convert_db: bool = False
    metric_labels: Dict[str, str] = field(default_factory=dict)
    split_by_sign: bool = False
    extra_options: Dict[str, Any] = field(default_factory=dict)
    write_forced_unconv_line: bool = False
