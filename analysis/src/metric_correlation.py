"""Shot-level metric comparison: Concordance Correlation Coefficient and scatter plots.

Two metrics (potentially from different decoders) are joined on shot_id so that
each scatter point corresponds to the same physical shot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import polars as pl

from analysis.src.config import PlotConfig
from analysis.src.data_manager import SimulationDataManager

# Columns used as join keys in addition to shot_id.
# shot_id is reset to 0 for each batch (.b8 file), so "batch" must be included
# to uniquely identify a shot across batches.
_CIRCUIT_PARAM_KEYS: list[str] = ["batch", "code", "d", "p", "rounds", "noisemodel", "xyz"]

# Metrics for which use_negative_gap is meaningful (logical-error shots get negated values).
_NEGATIVE_GAP_METRICS: frozenset[str] = frozenset({"logical_gap", "cluster_llr", "forced_gap_ml"})


# ---------------------------------------------------------------------------
# CCC computation
# ---------------------------------------------------------------------------

def concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Concordance Correlation Coefficient (Lin 1989) between *x* and *y*.

    CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

    Returns NaN when fewer than 2 finite pairs are available or the denominator
    is zero.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    mean_x, mean_y = x.mean(), y.mean()
    var_x = x.var(ddof=0)
    var_y = y.var(ddof=0)
    cov_xy = float(np.cov(x, y, ddof=0)[0, 1])
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return float(2.0 * cov_xy / denom) if denom > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MetricPairConfig:
    """Configuration for a shot-level scatter plot between two metrics.

    Parameters
    ----------
    x_metric :
        Metric name for the x-axis (e.g. ``"logical_gap"``).
    y_metric :
        Metric name for the y-axis (e.g. ``"forced_gap_ml"``).
    filters :
        Experiment parameter filters shared by both metrics.
    x_decoder_names :
        Restrict x-metric loading to these decoders. ``None`` includes all.
    y_decoder_names :
        Restrict y-metric loading to these decoders. ``None`` includes all.
    group_by :
        Circuit-param columns whose unique combinations produce separate
        scatter series (e.g. ``["d", "p"]``).
    batch_indices :
        Restrict to specific batch indices. ``None`` loads all batches.
    use_negative_gap :
        When ``True``, shots where the respective decoder made a logical error
        are plotted with their metric value negated (x errors negate x values,
        y errors negate y values).  Mirrors the convention in
        :class:`~analysis.src.analyzers.NumericMetricAnalyzer`.
    report_confidence_statistics :
        When ``True`` and the pair is ``logical_gap`` versus ``forced_gap_ml``,
        print a probability table for overconfident / underconfident shots and
        the same table split by logical-error status.
    """

    x_metric: str
    y_metric: str
    filters: dict[str, Any] = field(default_factory=dict)
    x_decoder_names: list[str] | None = None
    y_decoder_names: list[str] | None = None
    group_by: list[str] = field(default_factory=list)
    batch_indices: list[int] | None = None
    use_negative_gap: bool = False
    report_confidence_statistics: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_join(
    manager: SimulationDataManager,
    config: MetricPairConfig,
) -> pl.DataFrame:
    """Load both metrics and inner-join on shot_id + shared circuit params."""
    x_plot_cfg = PlotConfig(
        metric_name=config.x_metric,
        decoder_names=config.x_decoder_names,
        filters=config.filters,
        batch_indices=config.batch_indices,
    )
    y_plot_cfg = PlotConfig(
        metric_name=config.y_metric,
        decoder_names=config.y_decoder_names,
        filters=config.filters,
        batch_indices=config.batch_indices,
    )

    lf_x = manager.query(x_plot_cfg)
    lf_y = manager.query(y_plot_cfg)

    # The metric column name in the final parquet is just the metric name itself
    # (the "metric_" prefix is stripped by _collect_results in __main__.py).
    x_col = config.x_metric
    y_col = config.y_metric

    schema_x = set(lf_x.collect_schema().names())
    schema_y = set(lf_y.collect_schema().names())

    # Join on shot_id + whichever circuit-param columns exist in both schemas.
    join_keys = ["shot_id"] + [
        k for k in _CIRCUIT_PARAM_KEYS if k in schema_x and k in schema_y
    ]
    # Also keep group_by columns from the x side.
    extra_x = [c for c in config.group_by if c in schema_x and c not in join_keys]

    x_select = [c for c in (join_keys + extra_x + [x_col, "is_logical_error"]) if c in schema_x]
    y_select = [c for c in (join_keys + [y_col, "is_logical_error"]) if c in schema_y]

    df_x = lf_x.select(x_select).collect()
    df_y = lf_y.select(y_select).collect()

    # Rename is_logical_error before joining to avoid column name conflicts.
    if "is_logical_error" in df_x.columns:
        df_x = df_x.rename({"is_logical_error": "is_logical_error_x"})
    if "is_logical_error" in df_y.columns:
        df_y = df_y.rename({"is_logical_error": "is_logical_error_y"})

    actual_keys = [k for k in join_keys if k in df_x.columns and k in df_y.columns]
    return df_x.join(df_y, on=actual_keys, how="inner")


def _confidence_probability_table(part: pl.DataFrame) -> pl.DataFrame:
    """Return over/underconfidence probabilities overall and by logical error."""
    needed = ["logical_gap", "forced_gap_ml", "is_logical_error_x"]
    if not all(col in part.columns for col in needed):
        raise ValueError(
            "report_confidence_statistics requires logical_gap, forced_gap_ml, "
            "and is_logical_error_x to be present after joining"
        )

    sub = part.select(needed).drop_nulls(["logical_gap", "forced_gap_ml"])
    if sub.is_empty():
        return pl.DataFrame(
            [
                {
                    "subset": "overall",
                    "n": 0,
                    "p_overconfident": float("nan"),
                    "p_underconfident": float("nan"),
                    "p_equal": float("nan"),
                },
                {
                    "subset": "logical_error",
                    "n": 0,
                    "p_overconfident": float("nan"),
                    "p_underconfident": float("nan"),
                    "p_equal": float("nan"),
                },
                {
                    "subset": "non_logical_error",
                    "n": 0,
                    "p_overconfident": float("nan"),
                    "p_underconfident": float("nan"),
                    "p_equal": float("nan"),
                },
            ]
        )

    logical_vals = np.abs(sub["logical_gap"].to_numpy().astype(float))
    forced_vals = np.abs(sub["forced_gap_ml"].to_numpy().astype(float))
    logical_error = sub["is_logical_error_x"].to_numpy().astype(bool)

    over_mask = forced_vals > logical_vals
    under_mask = forced_vals < logical_vals
    equal_mask = ~(over_mask | under_mask)

    def _row(label: str, mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        if n == 0:
            return {
                "subset": label,
                "n": 0,
                "p_overconfident": float("nan"),
                "p_underconfident": float("nan"),
                "p_equal": float("nan"),
            }
        return {
            "subset": label,
            "n": n,
            "p_overconfident": float(over_mask[mask].mean()),
            "p_underconfident": float(under_mask[mask].mean()),
            "p_equal": float(equal_mask[mask].mean()),
        }

    return pl.DataFrame(
        [
            _row("overall", np.ones(sub.height, dtype=bool)),
            _row("logical_error", logical_error),
            _row("non_logical_error", ~logical_error),
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_metric_scatter(
    manager: SimulationDataManager,
    config: MetricPairConfig,
    *,
    ax: plt.Axes | None = None,
    scatter_kw: dict[str, Any] | None = None,
    print_ccc: bool = True,
) -> tuple[plt.Axes, dict[tuple, float]]:
    """Scatter plot of two metrics joined on shot_id with y=x reference line.

    Parameters
    ----------
    manager :
        Simulation data manager.
    config :
        Pair configuration.
    ax :
        Axes to draw on. A new figure is created when ``None``.
    scatter_kw :
        Extra keyword arguments forwarded to :func:`matplotlib.axes.Axes.scatter`.
    print_ccc :
        Print CCC value(s) to stdout.

    Returns
    -------
    ax :
        The axes containing the scatter plot.
    ccc_dict :
        Maps each partition key-tuple to its CCC value. Uses an empty tuple
        ``()`` when *config.group_by* is empty.
    """
    if ax is None:
        _, ax = plt.subplots()

    if config.report_confidence_statistics and (
        config.x_metric != "logical_gap" or config.y_metric != "forced_gap_ml"
    ):
        raise ValueError(
            "report_confidence_statistics is only supported when x_metric="
            "'logical_gap' and y_metric='forced_gap_ml'"
        )

    scatter_kw = dict(scatter_kw or {})
    x_col = config.x_metric
    y_col = config.y_metric

    df = _load_and_join(manager, config)
    ccc_dict: dict[tuple, float] = {}

    def _xy_from_part(part: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        needed = [x_col, y_col]
        if config.use_negative_gap:
            if "is_logical_error_x" in part.columns:
                needed.append("is_logical_error_x")
            if "is_logical_error_y" in part.columns:
                needed.append("is_logical_error_y")
        sub = part.select(needed).drop_nulls([x_col, y_col])
        x = sub[x_col].to_numpy().astype(float)
        y = sub[y_col].to_numpy().astype(float)
        if config.use_negative_gap:
            if x_col in _NEGATIVE_GAP_METRICS and "is_logical_error_x" in sub.columns:
                err_x = sub["is_logical_error_x"].to_numpy().astype(bool)
                x = x.copy()
                x[err_x] *= -1.0
            if y_col in _NEGATIVE_GAP_METRICS and "is_logical_error_y" in sub.columns:
                err_y = sub["is_logical_error_y"].to_numpy().astype(bool)
                y = y.copy()
                y[err_y] *= -1.0
        return x, y

    if not config.group_by:
        x, y = _xy_from_part(df)
        ccc = concordance_correlation_coefficient(x, y)
        ccc_dict[()] = ccc
        if print_ccc:
            print(f"CCC({config.x_metric}, {config.y_metric}) = {ccc:.6f}")
        ax.scatter(x, y, **scatter_kw)
        if config.report_confidence_statistics:
            print(
                f"Confidence probabilities for {config.x_metric} vs {config.y_metric}"
            )
            print(_confidence_probability_table(df))
    else:
        group_cols = [c for c in config.group_by if c in df.columns]
        partitions: dict[tuple, pl.DataFrame] = df.partition_by(
            group_cols, as_dict=True
        )
        for key_vals in sorted(partitions):
            x, y = _xy_from_part(partitions[key_vals])
            ccc = concordance_correlation_coefficient(x, y)
            key_tup = key_vals if isinstance(key_vals, tuple) else (key_vals,)
            ccc_dict[key_tup] = ccc
            label_str = ", ".join(
                f"{k}={v}" for k, v in zip(group_cols, key_tup)
            )
            if print_ccc:
                print(
                    f"CCC({config.x_metric}, {config.y_metric})"
                    f" [{label_str}] = {ccc:.6f}"
                )
            ax.scatter(x, y, label=f"{label_str} (CCC={ccc:.3f})", **scatter_kw)
            if config.report_confidence_statistics:
                print(
                    f"Confidence probabilities for {config.x_metric} vs {config.y_metric}"
                    f" [{label_str}]"
                )
                print(_confidence_probability_table(partitions[key_vals]))

    # y = x reference line spanning the data range (uses transformed values)
    x_all, y_all = _xy_from_part(df)
    if x_all.size > 0:
        all_vals = np.concatenate([x_all, y_all])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, zorder=0)

    return ax, ccc_dict


# ---------------------------------------------------------------------------
# 2D binned heatmaps: sample-count and conditional logical-error rate
# ---------------------------------------------------------------------------

@dataclass
class MetricHeatmapConfig:
    """Configuration for a shot-level 2D binned heatmap between two metrics.

    Uses the same shot-matched join as :class:`MetricPairConfig` (see
    :func:`_load_and_join`): ``x_metric`` and ``y_metric`` are inner-joined on
    ``shot_id`` (plus shared circuit-parameter columns), so each bin
    aggregates shots present in both.

    Parameters
    ----------
    x_metric :
        Metric name for the x-axis (e.g. ``"linearize_logicalgap"``).
    y_metric :
        Metric name for the y-axis (e.g. ``"logical_gap"``).
    filters :
        Experiment parameter filters shared by both metrics.
    x_decoder_names :
        Restrict x-metric loading to these decoders. ``None`` includes all.
    y_decoder_names :
        Restrict y-metric loading to these decoders. ``None`` includes all.
    batch_indices :
        Restrict to specific batch indices. ``None`` loads all batches.
    use_negative_gap :
        When ``True``, shots where the respective decoder made a logical
        error are binned with their metric value negated (same convention as
        :attr:`MetricPairConfig.use_negative_gap`); each axis is negated by
        its *own* decoder's error flag (``is_logical_error_x`` for ``x``,
        ``is_logical_error_y`` for ``y``), independent of ``error_source``.
    bins :
        Number of bins, either a single int (used for both axes) or an
        ``(nx, ny)`` pair. Forwarded to :func:`numpy.histogram2d`.
    x_range :
        Optional ``(min, max)`` bin range for the x-axis. ``None`` uses the
        observed data range.
    y_range :
        Optional ``(min, max)`` bin range for the y-axis. ``None`` uses the
        observed data range.
    error_source :
        Which side's ``is_logical_error`` to use for the conditional
        logical-error-rate heatmap (:func:`plot_metric_conditional_error_heatmap`).
        ``"x"`` (default) uses ``is_logical_error_x``, i.e. ``x_metric``'s own
        decoder/decode; ``"y"`` uses ``is_logical_error_y``, i.e.
        ``y_metric``'s. Has no effect on :func:`plot_metric_count_heatmap`.
    """

    x_metric: str
    y_metric: str
    filters: dict[str, Any] = field(default_factory=dict)
    x_decoder_names: list[str] | None = None
    y_decoder_names: list[str] | None = None
    batch_indices: list[int] | None = None
    use_negative_gap: bool = False
    bins: int | tuple[int, int] = 60
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    error_source: Literal["x", "y"] = "x"


@dataclass
class MetricHeatmapBins:
    """2D-binned per-shot counts for a metric pair (see :class:`MetricHeatmapConfig`)."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    #: counts[i, j] = number of shots with y in bin i, x in bin j (pcolormesh convention)
    counts: np.ndarray
    #: error_counts[i, j] = of those, how many have the config.error_source is_logical_error == True
    error_counts: np.ndarray
    n_total: int


def compute_metric_heatmap_bins(
    manager: SimulationDataManager,
    config: MetricHeatmapConfig,
) -> MetricHeatmapBins:
    """Load, join, and 2D-bin *config.x_metric* vs. *config.y_metric* per shot.

    ``error_counts`` reflects ``is_logical_error_x`` or ``is_logical_error_y``
    depending on ``config.error_source`` (see :class:`MetricHeatmapConfig`).

    Raises
    ------
    ValueError
        If no ``is_logical_error`` column tied to the requested
        ``error_source`` is found after joining, or if no shots remain after
        dropping nulls/non-finite values.
    """
    if config.error_source not in ("x", "y"):
        raise ValueError(f"error_source must be 'x' or 'y', got {config.error_source!r}")

    pair_config = MetricPairConfig(
        x_metric=config.x_metric,
        y_metric=config.y_metric,
        filters=config.filters,
        x_decoder_names=config.x_decoder_names,
        y_decoder_names=config.y_decoder_names,
        batch_indices=config.batch_indices,
    )
    df = _load_and_join(manager, pair_config)

    error_col = "is_logical_error_x" if config.error_source == "x" else "is_logical_error_y"
    error_metric = config.x_metric if config.error_source == "x" else config.y_metric
    if error_col not in df.columns:
        raise ValueError(
            f"conditional logical-error heatmap requires an is_logical_error "
            f"column tied to error_source={config.error_source!r} "
            f"(metric='{error_metric}'), but none was found after joining. "
            f"Check {'x' if config.error_source == 'x' else 'y'}_decoder_names "
            "and that its directory has logicalerror data."
        )

    needed = list(
        dict.fromkeys(
            [config.x_metric, config.y_metric, error_col]
            + (["is_logical_error_x", "is_logical_error_y"] if config.use_negative_gap else [])
        )
    )
    needed = [c for c in needed if c in df.columns]
    sub = df.select(needed).drop_nulls([config.x_metric, config.y_metric])

    x = sub[config.x_metric].to_numpy().astype(float)
    y = sub[config.y_metric].to_numpy().astype(float)
    is_error = sub[error_col].to_numpy().astype(bool)

    if config.use_negative_gap:
        if config.x_metric in _NEGATIVE_GAP_METRICS and "is_logical_error_x" in sub.columns:
            err_x = sub["is_logical_error_x"].to_numpy().astype(bool)
            x = x.copy()
            x[err_x] *= -1.0
        if config.y_metric in _NEGATIVE_GAP_METRICS and "is_logical_error_y" in sub.columns:
            err_y = sub["is_logical_error_y"].to_numpy().astype(bool)
            y = y.copy()
            y[err_y] *= -1.0

    finite = np.isfinite(x) & np.isfinite(y)
    x, y, is_error = x[finite], y[finite], is_error[finite]

    if x.size == 0:
        raise ValueError(
            "No shots remain after joining/dropping nulls and non-finite "
            f"values for x_metric='{config.x_metric}', y_metric='{config.y_metric}' "
            f"(filters={config.filters})."
        )

    bins = config.bins if isinstance(config.bins, tuple) else (config.bins, config.bins)
    hist_range = None
    if config.x_range is not None or config.y_range is not None:
        xr = config.x_range or (float(x.min()), float(x.max()))
        yr = config.y_range or (float(y.min()), float(y.max()))
        hist_range = [list(xr), list(yr)]

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=hist_range)
    error_counts, _, _ = np.histogram2d(x[is_error], y[is_error], bins=[x_edges, y_edges])

    # np.histogram2d returns shape (nx, ny); transpose to the (ny, nx) shape
    # expected by ax.pcolormesh(x_edges, y_edges, C).
    return MetricHeatmapBins(
        x_edges=x_edges,
        y_edges=y_edges,
        counts=counts.T,
        error_counts=error_counts.T,
        n_total=int(x.size),
    )


def plot_metric_count_heatmap(
    manager: SimulationDataManager,
    config: MetricHeatmapConfig,
    ax: plt.Axes,
    *,
    cmap: str = "viridis",
    add_colorbar: bool = True,
    colorbar_label: str = r"$\log_{10} N$",
) -> plt.Axes:
    """Draw a heatmap of ``log10(N(x, y))``: where the samples are.

    Bins with zero shots are left blank (masked white).
    """
    heat = compute_metric_heatmap_bins(manager, config)
    counts = heat.counts

    log_counts = np.full(counts.shape, np.nan, dtype=float)
    positive = counts > 0
    log_counts[positive] = np.log10(counts[positive])

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    mesh = ax.pcolormesh(
        heat.x_edges, heat.y_edges, np.ma.masked_invalid(log_counts),
        cmap=cmap_obj, shading="auto",
    )
    ax.set_xlabel(config.x_metric)
    ax.set_ylabel(config.y_metric)
    if add_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax)
        colorbar.set_label(colorbar_label)
    return ax


def plot_metric_conditional_error_heatmap(
    manager: SimulationDataManager,
    config: MetricHeatmapConfig,
    ax: plt.Axes,
    *,
    cmap: str = "magma",
    add_colorbar: bool = True,
    colorbar_label: str | None = None,
    log_color: bool = True,
) -> plt.Axes:
    """Draw a heatmap of ``P(is_logical_error = 1 | x, y)``: where it's dangerous.

    The logical-error flag comes from ``config.error_source`` ("x" or "y",
    see :class:`MetricHeatmapConfig`) -- i.e. whichever of ``x_metric`` /
    ``y_metric``'s own decoder is chosen decides what counts as an error.

    Bins with zero shots are left blank (masked white); with ``log_color=True``
    (default), bins with zero observed errors are also left blank since a
    rate of exactly 0 cannot be placed on a log color scale.
    """
    heat = compute_metric_heatmap_bins(manager, config)

    if colorbar_label is None:
        error_metric = config.x_metric if config.error_source == "x" else config.y_metric
        colorbar_label = rf"$P(\mathrm{{logical\ error}}=1\mid x,y)$ [{error_metric}]"
    counts = heat.counts

    with np.errstate(invalid="ignore", divide="ignore"):
        ler = heat.error_counts / counts
    ler[counts == 0] = np.nan

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    norm = None
    vmin, vmax = None, None
    if log_color:
        positive = ler[np.isfinite(ler) & (ler > 0)]
        if positive.size == 0:
            raise ValueError(
                "No bins with a positive conditional logical-error rate to "
                "plot; pass log_color=False for a linear 0-1 color scale."
            )
        vmin = float(positive.min())
        vmax = float(positive.max())
        if vmin == vmax:
            vmin = max(vmin / 10.0, np.nextafter(0.0, 1.0))
            vmax = vmax * 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = 0.0, 1.0

    mesh = ax.pcolormesh(
        heat.x_edges, heat.y_edges, np.ma.masked_invalid(ler),
        cmap=cmap_obj, norm=norm, vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None, shading="auto",
    )
    ax.set_xlabel(config.x_metric)
    ax.set_ylabel(config.y_metric)
    if add_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax)
        colorbar.set_label(colorbar_label)
    return ax
