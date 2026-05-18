"""Metric analyzer classes for QEC simulation result visualisation.

All plot methods render **scatter plots** (not histograms or bar charts).
Binned modes compute per-bin proportions with Wilson confidence intervals
(from :mod:`analysis.src.confidence`) drawn as shaded bands.

Classes
-------
NumericMetricAnalyzer
    Scatter plot of a continuous metric's distribution.
BooleanMetricAnalyzer
    Scatter plot of accept rates for a binary metric.
ConditionalLERAnalyzer
    Scatter plot of P(logical error | metric value) with optional log-odds
    linear fitting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from analysis.src.config import ConditionalLERConfig, PlotConfig
from analysis.src.confidence import (
    BinnedProportions,
    bin_density,
    bin_proportions,
    shade_ci,
    wilson_ci,
)

# plot_params keys consumed internally; not forwarded to matplotlib primitives.
_INTERNAL_PLOT_PARAMS = frozenset(
    {"bins", "alpha_ci", "shade_alpha", "round_digits", "use_negative_gap"}
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_label(partition_keys: list[str], key_vals: tuple[Any, ...]) -> str:
    return ", ".join(f"{k}={v}" for k, v in zip(partition_keys, key_vals))


def _effective_partition_keys(config: PlotConfig) -> list[str]:
    keys = list(config.group_by)
    if config.separate_logical_error and "is_logical_error" not in keys:
        keys.append("is_logical_error")
    return keys


def _collect_for_plot(
    lf: pl.LazyFrame, metric_name: str, partition_keys: list[str]
) -> pl.DataFrame:
    needed = list(dict.fromkeys([metric_name, "is_logical_error"] + partition_keys))
    schema_names = set(lf.collect_schema().names())
    existing = [c for c in needed if c in schema_names]
    return lf.select(existing).collect()


def _scatter_color(artist) -> Any:
    """Extract the face colour from a PathCollection returned by ax.scatter()."""
    return artist.get_facecolor()[0]


def _normalize_round_digits(round_digits: Any) -> int | None:
    if round_digits is None:
        return None
    if isinstance(round_digits, bool):
        raise ValueError("round_digits must be an integer or None")
    if isinstance(round_digits, int):
        return round_digits
    if isinstance(round_digits, float) and round_digits.is_integer():
        return int(round_digits)
    raise ValueError("round_digits must be an integer or None")


def _normalize_bins(bins: Any) -> int | None:
    if bins is None:
        return None
    if isinstance(bins, bool):
        return None
    if isinstance(bins, int):
        return bins if bins > 0 else None
    if isinstance(bins, float) and bins.is_integer():
        return int(bins) if bins > 0 else None
    raise ValueError("bins must be a positive integer or None")


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class AbstractMetricAnalyzer(ABC):
    """Base class for per-metric plotting strategies."""

    @abstractmethod
    def plot_distribution(
        self,
        lf: pl.LazyFrame,
        config: PlotConfig,
        ax: plt.Axes,
    ) -> None:
        """Render the distribution of the metric described by *config* onto *ax*.

        Parameters
        ----------
        lf:
            LazyFrame produced by :meth:`SimulationDataManager.query`.
        config:
            Plotting configuration.
        ax:
            Matplotlib axes to draw on.
        """


# ---------------------------------------------------------------------------
# Numeric metrics (e.g. logical_gap)
# ---------------------------------------------------------------------------

class NumericMetricAnalyzer(AbstractMetricAnalyzer):
    """Scatter plots for continuous / numeric metrics.

    **Binned mode** (``plot_params["bins"]`` is a positive integer):
        The metric range is divided into equal-width bins.  Each bin is one
        scatter point whose y-value is the **frequency** (shot count) in that
        bin.  Wilson CIs computed on the underlying proportion are converted to
        the count scale and shaded around each point.

    **Non-binned mode** (``"bins"`` absent or non-positive):
        A scatter plot of per-value frequencies (x = metric value,
        y = count).  No CI is drawn in this mode.

    ``plot_params`` keys recognised by this class:

    * ``"bins"`` – number of bins (positive int); triggers binned mode.
    * ``"alpha_ci"`` – Wilson CI significance level (default 0.05).
    * ``"shade_alpha"`` – opacity of the CI shade (default 0.2).
    * ``"round_digits"`` – decimal digits for rounding (None disables).
    * ``"use_negative_gap"`` – flip logical-error shots negative (bool).

    All other ``plot_params`` keys are forwarded to :func:`matplotlib.axes.Axes.scatter`.
    """

    def plot_distribution(
        self,
        lf: pl.LazyFrame,
        config: PlotConfig,
        ax: plt.Axes,
    ) -> None:
        partition_keys = _effective_partition_keys(config)
        df = _collect_for_plot(lf, config.metric_name, partition_keys)

        bins: int | None = _normalize_bins(config.plot_params.get("bins"))
        alpha_ci: float = config.plot_params.get("alpha_ci", 0.05)
        shade_alpha: float = config.plot_params.get("shade_alpha", 0.2)
        round_digits = _normalize_round_digits(config.plot_params.get("round_digits"))
        use_negative_gap = bool(config.plot_params.get("use_negative_gap", False))
        scatter_kw = {k: v for k, v in config.plot_params.items()
                      if k not in _INTERNAL_PLOT_PARAMS}

        if use_negative_gap and config.metric_name not in {
            "logical_gap",
            "linearize_logicalgap",
        }:
            raise ValueError(
                "use_negative_gap is only supported for logical_gap and "
                "linearize_logicalgap"
            )

        if not partition_keys:
            values = self._extract_values(
                df, config.metric_name, round_digits, use_negative_gap
            )
            self._plot_group(ax, values, bins, alpha_ci, shade_alpha, scatter_kw)
        else:
            partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                partition_keys, as_dict=True
            )
            for key_vals in sorted(partitions):
                values = self._extract_values(
                    partitions[key_vals], config.metric_name, round_digits, use_negative_gap
                )
                label = _make_label(partition_keys, key_vals)
                self._plot_group(ax, values, bins, alpha_ci, shade_alpha, scatter_kw,
                                 label=label)
            ax.legend()

        ax.set_xlabel(config.metric_name)
        ax.set_ylabel("Frequency")

    def list_unique_values(
        self,
        lf: pl.LazyFrame,
        config: PlotConfig,
    ) -> list:
        """Return a sorted list of unique metric values (debug helper).

        Parameters
        ----------
        lf:
            LazyFrame produced by :meth:`SimulationDataManager.query`.
        config:
            Only ``config.metric_name`` is used.

        Returns
        -------
        list
            Sorted unique values of the metric column (nulls excluded).
        """
        round_digits = _normalize_round_digits(config.plot_params.get("round_digits"))
        use_negative_gap = bool(config.plot_params.get("use_negative_gap", False))

        if use_negative_gap and config.metric_name not in {
            "logical_gap",
            "linearize_logicalgap",
        }:
            raise ValueError(
                "use_negative_gap is only supported for logical_gap and "
                "linearize_logicalgap"
            )

        df = _collect_for_plot(lf, config.metric_name, [])
        values = self._extract_values(
            df, config.metric_name, round_digits, use_negative_gap
        )
        if values.size == 0:
            return []

        return np.unique(values).tolist()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _plot_group(
        self,
        ax: plt.Axes,
        values: np.ndarray,
        bins: int | None,
        alpha_ci: float,
        shade_alpha: float,
        scatter_kw: dict,
        label: str | None = None,
    ) -> None:
        if values.size == 0:
            return

        if bins is not None:
            n_total = len(values)
            bstats: BinnedProportions = bin_density(values, bins=bins, alpha=alpha_ci)
            valid = ~np.isnan(bstats.proportions)
            # y-axis: raw frequency (count); CI converted to the same scale
            freqs = bstats.counts.astype(float)
            ci_low_count = bstats.ci_low * n_total
            ci_high_count = bstats.ci_high * n_total
            sc = ax.scatter(
                bstats.centers[valid], freqs[valid],
                label=label, **scatter_kw,
            )
            shade_ci(ax, bstats.centers, ci_low_count, ci_high_count,
                     color=_scatter_color(sc), alpha=shade_alpha)
        else:
            # Per-value frequency — no CI needed in non-binned mode
            unique_vals, counts = np.unique(values, return_counts=True)
            ax.scatter(unique_vals, counts, label=label, **scatter_kw)

    def _extract_values(
        self,
        df: pl.DataFrame,
        metric_name: str,
        round_digits: int | None,
        use_negative_gap: bool,
    ) -> np.ndarray:
        if use_negative_gap:
            if "is_logical_error" not in df.columns:
                raise ValueError(
                    "is_logical_error column is required when use_negative_gap is True"
                )
            sub = df.select([metric_name, "is_logical_error"]).drop_nulls(
                [metric_name, "is_logical_error"]
            )
            values = sub[metric_name].to_numpy().astype(float)
            is_error = sub["is_logical_error"].to_numpy().astype(bool)
            if is_error.any():
                values = values.copy()
                values[is_error] *= -1.0
        else:
            values = df[metric_name].drop_nulls().to_numpy().astype(float)

        if round_digits is not None:
            values = np.round(values, decimals=round_digits)

        return values


# ---------------------------------------------------------------------------
# Boolean metrics (e.g. ar-lec accept/reject)
# ---------------------------------------------------------------------------

class BooleanMetricAnalyzer(AbstractMetricAnalyzer):
    """Scatter plots of accept rates for binary / boolean metrics.

    One scatter point is drawn per unique combination of ``config.group_by``
    values (and ``is_logical_error`` when ``config.separate_logical_error`` is
    ``True``).  Wilson CIs are displayed as error bars.

    ``plot_params`` keys recognised by this class:

    * ``"alpha_ci"`` – Wilson CI significance level (default 0.05).

    All other ``plot_params`` keys except ``"bins"`` are forwarded to
    :func:`matplotlib.axes.Axes.errorbar`.
    """

    def plot_distribution(
        self,
        lf: pl.LazyFrame,
        config: PlotConfig,
        ax: plt.Axes,
    ) -> None:
        partition_keys = _effective_partition_keys(config)
        df = _collect_for_plot(lf, config.metric_name, partition_keys)

        alpha_ci: float = config.plot_params.get("alpha_ci", 0.05)
        errorbar_kw = {k: v for k, v in config.plot_params.items()
                       if k not in _INTERNAL_PLOT_PARAMS}

        col = df[config.metric_name]
        accept_series = col if col.dtype == pl.Boolean else (col > 0.5)

        if not partition_keys:
            n = len(accept_series)
            k = int(accept_series.sum())
            rate = k / n if n > 0 else 0.0
            ci_low, ci_high = (wilson_ci(k, n, alpha=alpha_ci) if n > 0
                               else (0.0, 0.0))
            ax.errorbar(
                [0], [rate],
                yerr=[[rate - float(ci_low)], [float(ci_high) - rate]],
                fmt="o", capsize=5, **errorbar_kw,
            )
            ax.set_xticks([0])
            ax.set_xticklabels(["all"])
            ax.set_ylabel("Accept rate")
            ax.set_ylim(0, 1)
            return

        partitions: dict[tuple, pl.DataFrame] = df.partition_by(
            partition_keys, as_dict=True
        )

        labels: list[str] = []
        rates: list[float] = []
        ci_lows: list[float] = []
        ci_highs: list[float] = []

        for key_vals in sorted(partitions):
            part_col = partitions[key_vals][config.metric_name]
            part_accept = part_col if part_col.dtype == pl.Boolean else (part_col > 0.5)
            n = len(part_accept)
            k = int(part_accept.sum())
            rate = k / n if n > 0 else 0.0
            ci_low, ci_high = (wilson_ci(k, n, alpha=alpha_ci) if n > 0
                               else (0.0, 0.0))

            labels.append(_make_label(partition_keys, key_vals))
            rates.append(rate)
            ci_lows.append(float(ci_low))
            ci_highs.append(float(ci_high))

        x_arr = np.arange(len(labels), dtype=float)
        rates_arr = np.array(rates)
        low_arr = np.array(ci_lows)
        high_arr = np.array(ci_highs)

        ax.errorbar(
            x_arr, rates_arr,
            yerr=[rates_arr - low_arr, high_arr - rates_arr],
            fmt="o", capsize=5, **errorbar_kw,
        )
        ax.set_xticks(x_arr)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Accept rate")
        ax.set_ylim(0, 1)


# ---------------------------------------------------------------------------
# Conditional logical error rate
# ---------------------------------------------------------------------------

class ConditionalLERAnalyzer:
    """Conditional logical error rate (LER) scatter plots.

    Computes P(logical_error | metric ∈ bin) for uniformly spaced bins along
    the metric axis.  Wilson CIs are shaded.

    Supported metrics (see :data:`~analysis.src.config.CONDITIONAL_LER_SUPPORTED_METRICS`):

    * ``"logical_gap"``
    * ``"linearize_logicalgap"``

    Methods
    -------
    plot_conditional_ler(lf, config, ax)
        Main LER scatter with CI shading.
    plot_fitting(lf, config, ax)
        Log-odds scatter with linear fit (call when ``config.get_fitting_plot``
        is ``True``).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_conditional_ler(
        self,
        lf: pl.LazyFrame,
        config: ConditionalLERConfig,
        ax: plt.Axes,
    ) -> None:
        """Draw P(logical error | metric) scatter with Wilson CI shading.

        Parameters
        ----------
        lf:
            LazyFrame from :meth:`SimulationDataManager.query`.  Must contain
            ``config.metric_name`` and ``"is_logical_error"`` columns.
        config:
            Conditional LER configuration.
        ax:
            Axes to draw on.
        """
        df = self._collect(lf, config)

        if not config.group_by:
            self._plot_ler_group(ax, df, config)
        else:
            partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                config.group_by, as_dict=True
            )
            for key_vals in sorted(partitions):
                label = _make_label(config.group_by, key_vals)
                self._plot_ler_group(ax, partitions[key_vals], config, label=label)
            ax.legend()

        ax.set_xlabel(config.metric_name)
        ax.set_ylabel(f"P(logical error | {config.metric_name})")

    def plot_fitting(
        self,
        lf: pl.LazyFrame,
        config: ConditionalLERConfig,
        ax: plt.Axes,
    ) -> None:
        """Draw the log-odds scatter and linear fit.

        The y-axis shows ``z = log((1 - y) / y)`` where ``y`` is the per-bin
        conditional LER.  A least-squares linear fit ``z = k·g + l`` is
        overlaid as a dotted line; ``k`` and ``l`` appear in the legend.

        This method is a no-op when ``config.get_fitting_plot`` is ``False``.

        Parameters
        ----------
        lf, config, ax:
            Same as :meth:`plot_conditional_ler`.
        """
        if not config.get_fitting_plot:
            return

        df = self._collect(lf, config)

        if not config.group_by:
            self._plot_fitting_group(ax, df, config, label=None)
        else:
            partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                config.group_by, as_dict=True
            )
            for key_vals in sorted(partitions):
                label = _make_label(config.group_by, key_vals)
                self._plot_fitting_group(ax, partitions[key_vals], config, label=label)
            ax.legend()

        ax.set_xlabel(config.metric_name)
        ax.set_ylabel(r"$\log\!\left(\dfrac{1 - y}{y}\right)$")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect(self, lf: pl.LazyFrame, config: ConditionalLERConfig) -> pl.DataFrame:
        needed = list(dict.fromkeys(
            [config.metric_name, "is_logical_error"] + config.group_by
        ))
        schema_names = set(lf.collect_schema().names())
        existing = [c for c in needed if c in schema_names]
        return lf.select(existing).collect()

    def _metric_and_error(
        self, df: pl.DataFrame, config: ConditionalLERConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned (metric_values, is_logical_error) arrays, nulls dropped."""
        sub = df.drop_nulls([config.metric_name, "is_logical_error"])
        x = sub[config.metric_name].to_numpy().astype(float)
        y = sub["is_logical_error"].to_numpy().astype(float)
        round_digits = _normalize_round_digits(config.round_digits)
        if round_digits is not None:
            x = np.round(x, decimals=round_digits)
        return x, y

    def _plot_ler_group(
        self,
        ax: plt.Axes,
        df: pl.DataFrame,
        config: ConditionalLERConfig,
        label: str | None = None,
    ) -> None:
        x, y = self._metric_and_error(df, config)
        if x.size == 0:
            return

        bstats = self._ler_stats(x, y, config)
        valid = ~np.isnan(bstats.proportions)

        sc = ax.scatter(
            bstats.centers[valid], bstats.proportions[valid],
            label=label, s=30,
        )
        shade_ci(
            ax, bstats.centers, bstats.ci_low, bstats.ci_high,
            color=_scatter_color(sc), alpha=0.2,
        )

    def _plot_fitting_group(
        self,
        ax: plt.Axes,
        df: pl.DataFrame,
        config: ConditionalLERConfig,
        label: str | None = None,
    ) -> None:
        x, y = self._metric_and_error(df, config)
        if x.size == 0:
            return

        bstats = self._ler_stats(x, y, config)
        p = bstats.proportions

        # z = log((1-p)/p) is defined only for 0 < p < 1
        valid = (~np.isnan(p)) & (p > 0.0) & (p < 1.0)
        g = bstats.centers[valid]
        z = np.log((1.0 - p[valid]) / p[valid])

        if g.size < 2:
            return

        k, l = np.polyfit(g, z, 1)

        data_label = f"data ({label})" if label else "data"
        sc = ax.scatter(g, z, label=data_label, s=30)
        color = _scatter_color(sc)

        # Shade CI in z-space using Wilson CI on p
        ci_low = bstats.ci_low[valid]
        ci_high = bstats.ci_high[valid]
        ci_valid = (~np.isnan(ci_low)) & (~np.isnan(ci_high)) & (ci_low > 0.0) & (ci_high < 1.0)
        if ci_valid.any():
            g_ci = g[ci_valid]
            ci_low_sel = ci_low[ci_valid]
            ci_high_sel = ci_high[ci_valid]
            # f(p) = log((1-p)/p) is decreasing, so swap order for bounds
            z_low = np.log((1.0 - ci_high_sel) / ci_high_sel)
            z_high = np.log((1.0 - ci_low_sel) / ci_low_sel)
            shade_ci(ax, g_ci, z_low, z_high, color=color, alpha=0.2)

        g_line = np.linspace(float(g.min()), float(g.max()), 300)
        fit_label = (
            f"fit ({label + ', ' if label else ''}k={k:.3f}, l={l:.3f})"
        )
        ax.plot(g_line, k * g_line + l, linestyle="--", color=color, label=fit_label)

    def _ler_stats(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: ConditionalLERConfig,
    ) -> BinnedProportions:
        if config.bins is None:
            return self._value_proportions(x, y, alpha=config.alpha)
        return bin_proportions(x, y, bins=config.bins, alpha=config.alpha)

    def _value_proportions(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alpha: float,
    ) -> BinnedProportions:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.size == 0:
            empty: np.ndarray = np.array([], dtype=float)
            return BinnedProportions(
                empty, empty, empty, empty, np.array([], dtype=int), np.array([], dtype=int)
            )

        unique_vals, inverse = np.unique(x, return_inverse=True)
        n_groups = len(unique_vals)

        counts = np.zeros(n_groups, dtype=int)
        totals = np.zeros(n_groups, dtype=int)
        for i in range(n_groups):
            mask = inverse == i
            totals[i] = int(mask.sum())
            counts[i] = int(y[mask].sum())

        proportions = np.where(totals > 0, counts / totals, np.nan)
        ci_low = np.full(n_groups, np.nan)
        ci_high = np.full(n_groups, np.nan)
        valid = totals > 0
        if valid.any():
            low, high = wilson_ci(counts[valid], totals[valid], alpha=alpha)
            ci_low[valid] = low
            ci_high[valid] = high

        return BinnedProportions(
            centers=unique_vals,
            proportions=proportions,
            ci_low=ci_low,
            ci_high=ci_high,
            counts=counts,
            totals=totals,
        )
