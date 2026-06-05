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
        ``"use_negative_gap"``, ``"convert_db"``.
    extra_options:
        Reserved for future extensibility (e.g. conditional LER, post-selection).
    batch_indices:
        Restrict loading to specific batch indices (1-based integers matching the
        ``batch=N`` suffix in parquet filenames). ``None`` loads all available batches.
    metric_labels:
        Mapping from internal metric name to a human-readable display label used in
        legend entries and axis labels (e.g. ``{"linearize_logicalgap": "BP+OSD"}``).
        Metric names absent from the mapping fall back to the raw name.
    xlabel:
        Override for the x-axis label.  When ``None`` (default) the label is derived
        automatically from the metric name / ``metric_labels``.
    ylabel:
        Override for the y-axis label.  When ``None`` (default) the label is
        ``"Frequency"``.
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
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None


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
        legend entries and axis labels (e.g. ``{"logical_gap": "BP decoder"}``).
        Metric names absent from the mapping fall back to the raw name.
    xlabel:
        Override for the x-axis label.  When ``None`` the label is derived
        automatically from the metric name / ``metric_labels``.
    ylabel:
        Override for the y-axis label.  When ``None`` the default label is used.
    """

    metric_name: str | List[str]
    decoder_names: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    bins: Optional[int] = None
    alpha: float = 0.05
    get_fitting_plot: bool = False
    batch_indices: Optional[List[int]] = None
    round_digits: Optional[int] = None
    convert_db: bool = False
    metric_labels: Dict[str, str] = field(default_factory=dict)
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
