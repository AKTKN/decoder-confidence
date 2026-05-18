from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Metrics supported by :class:`ConditionalLERAnalyzer`.
CONDITIONAL_LER_SUPPORTED_METRICS: frozenset[str] = frozenset(
    {"logical_gap", "linearize_logicalgap"}
)


@dataclass
class PlotConfig:
    """Encapsulates all configuration for a metric distribution plot.

    Parameters
    ----------
    metric_name:
        Name of the metric column to analyse (e.g. ``"logical_gap"``, ``"ar-lec"``).
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
        (e.g. ``{"bins": 50, "alpha": 0.7}``).
    extra_options:
        Reserved for future extensibility (e.g. conditional LER, post-selection).
    batch_indices:
        Restrict loading to specific batch indices (1-based integers matching the
        ``batch=N`` suffix in parquet filenames). ``None`` loads all available batches.
    """

    metric_name: str
    decoder_names: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    separate_logical_error: bool = False
    plot_params: Dict[str, Any] = field(default_factory=dict)
    extra_options: Dict[str, Any] = field(default_factory=dict)
    batch_indices: Optional[List[int]] = None


@dataclass
class ConditionalLERConfig:
    """Configuration for :class:`~analysis.src.analyzers.ConditionalLERAnalyzer`.

    Parameters
    ----------
    metric_name:
        Continuous metric used as the x-axis (e.g. ``"logical_gap"``,
        ``"linearize_logicalgap"``).  Must be one of
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
    """

    metric_name: str
    decoder_names: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    bins: Optional[int] = None
    alpha: float = 0.05
    get_fitting_plot: bool = False
    batch_indices: Optional[List[int]] = None
    round_digits: Optional[int] = None
