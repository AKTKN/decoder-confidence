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
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from analysis.src.config import (
    CONDITIONAL_LER_SUPPORTED_METRICS,
    ConditionalLERConfig,
    PlotConfig,
    normalize_metric_names,
)
from analysis.src.data_manager import SimulationDataManager
from analysis.src.confidence import (
    BinnedProportions,
    bin_density,
    bin_proportions,
    shade_ci,
    wilson_ci,
)

# plot_params keys consumed internally; not forwarded to matplotlib primitives.
_INTERNAL_PLOT_PARAMS = frozenset(
    {"bins", "alpha_ci", "shade_alpha", "round_digits", "use_negative_gap", "convert_db"}
)

# Metrics for which gap → dB conversion is meaningful.
_GAP_METRICS = frozenset({"logical_gap", "linearize_logicalgap", "forced_gap_ml"})

# Multiplicative factor for gap → dB: gap_dB = _DB_FACTOR * gap
_DB_FACTOR: float = 10.0 / np.log(10.0)


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


def _normalize_per_round(per_round: Any) -> int | None:
    if per_round is None:
        return None
    if isinstance(per_round, bool):
        raise ValueError("per_round must be an integer >= 1 or None")
    if isinstance(per_round, int) and per_round >= 1:
        return per_round
    raise ValueError("per_round must be an integer >= 1 or None")


def _logical_error_rate_from_lf(
    lf: pl.LazyFrame,
    per_round: int | None,
    return_num_shots: bool,
) -> float | tuple[float, int]:
    col = pl.col("is_logical_error").cast(pl.Int64)
    stats = lf.select(
        col.sum().alias("n_err"),
        pl.len().alias("n_total"),
    ).collect()
    if stats.height == 0:
        n_err = 0
        n_total = 0
    else:
        raw_err = stats["n_err"][0]
        raw_total = stats["n_total"][0]
        n_err = 0 if raw_err is None else int(raw_err)
        n_total = 0 if raw_total is None else int(raw_total)

    ler = (n_err / n_total) if n_total > 0 else 0.0
    if per_round is not None:
        ler /= per_round

    if return_num_shots:
        return ler, n_total
    return ler


def _metric_style_map(metric_names: list[str], ax: plt.Axes) -> dict[str, dict[str, Any]]:
    colors = plt.rcParams.get("axes.prop_cycle", None)
    if colors is not None:
        palette = colors.by_key().get("color", [])
    else:
        palette = []
    if not palette:
        palette = [f"C{i}" for i in range(10)]

    markers = ["o", "x", "^", "D", "v", ">", "<", "P", "X", "*"]

    style_map: dict[str, dict[str, Any]] = {}
    for i, metric in enumerate(metric_names):
        style_map[metric] = {
            "color": palette[i % len(palette)],
            "marker": markers[i % len(markers)],
        }
    return style_map


def _metric_label(metric_name: str, label: str | None) -> str:
    if label:
        return f"{metric_name} | {label}"
    return metric_name


def _validate_conditional_metric(metric_name: str) -> None:
    if metric_name not in CONDITIONAL_LER_SUPPORTED_METRICS:
        supported = ", ".join(sorted(CONDITIONAL_LER_SUPPORTED_METRICS))
        raise ValueError(
            f"metric_name '{metric_name}' is not supported for conditional LER. "
            f"Supported metrics: {supported}"
        )


# ---------------------------------------------------------------------------
# Post-selection result
# ---------------------------------------------------------------------------

@dataclass
class PostselectionResult:
    """Statistics from post-selection at a fixed metric threshold.

    Attributes
    ----------
    abort_rate:
        Fraction of shots aborted (discarded) by the threshold.
    post_ler:
        Logical error rate measured on the accepted (kept) shots.
    original_ler:
        Logical error rate without any post-selection.
    reduction_rate:
        ``post_ler / original_ler`` — values < 1 mean improvement.
    n_accepted:
        Number of shots kept after post-selection.
    n_total:
        Total number of shots before post-selection.
    ci_low:
        Wilson confidence-interval lower bound for *post_ler*.
    ci_high:
        Wilson confidence-interval upper bound for *post_ler*.
    """

    abort_rate: float
    post_ler: float
    original_ler: float
    reduction_rate: float
    n_accepted: int
    n_total: int
    ci_low: float
    ci_high: float

    def __repr__(self) -> str:
        def _fmt(v: float) -> str:
            return f"{v:.6g}" if not (v != v) else "nan"  # noqa: PLR0124

        return (
            f"PostselectionResult(\n"
            f"  abort_rate={_fmt(self.abort_rate)},\n"
            f"  post_ler={_fmt(self.post_ler)},\n"
            f"  original_ler={_fmt(self.original_ler)},\n"
            f"  reduction_rate={_fmt(self.reduction_rate)},\n"
            f"  n_accepted={self.n_accepted},\n"
            f"  n_total={self.n_total},\n"
            f"  ci_low={_fmt(self.ci_low)},\n"
            f"  ci_high={_fmt(self.ci_high)},\n"
            f")"
        )


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
        metric_names = normalize_metric_names(config.metric_name)
        multi_metric = len(metric_names) > 1
        partition_keys = _effective_partition_keys(config)

        bins: int | None = _normalize_bins(config.plot_params.get("bins"))
        alpha_ci: float = config.plot_params.get("alpha_ci", 0.05)
        shade_alpha: float = config.plot_params.get("shade_alpha", 0.2)
        round_digits = _normalize_round_digits(config.plot_params.get("round_digits"))
        use_negative_gap = bool(config.plot_params.get("use_negative_gap", False))
        convert_db = bool(config.plot_params.get("convert_db", False))

        if use_negative_gap:
            invalid = [n for n in metric_names if n not in _GAP_METRICS]
            if invalid:
                raise ValueError(
                    "use_negative_gap is only supported for gap metrics: "
                    + ", ".join(sorted(_GAP_METRICS))
                )
        if convert_db:
            invalid = [n for n in metric_names if n not in _GAP_METRICS]
            if invalid:
                raise ValueError(
                    "convert_db is only supported for gap metrics: "
                    + ", ".join(sorted(_GAP_METRICS))
                )

        style_map = _metric_style_map(metric_names, ax) if multi_metric else {}

        for metric_name in metric_names:
            df = _collect_for_plot(lf, metric_name, partition_keys)
            display_name = config.metric_labels.get(metric_name, metric_name)

            scatter_kw = {k: v for k, v in config.plot_params.items()
                          if k not in _INTERNAL_PLOT_PARAMS}
            if multi_metric:
                scatter_kw.pop("color", None)
                scatter_kw.pop("c", None)
                scatter_kw.pop("marker", None)
                scatter_kw.update(style_map[metric_name])

            if not partition_keys:
                label = display_name if multi_metric else None
                values = self._extract_values(
                    df, metric_name, round_digits, use_negative_gap, convert_db
                )
                self._plot_group(
                    ax, values, bins, alpha_ci, shade_alpha, scatter_kw, label=label
                )
            else:
                partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                    partition_keys, as_dict=True
                )
                for key_vals in sorted(partitions):
                    values = self._extract_values(
                        partitions[key_vals], metric_name, round_digits,
                        use_negative_gap, convert_db
                    )
                    label = _make_label(partition_keys, key_vals)
                    if multi_metric:
                        label = _metric_label(display_name, label)
                    self._plot_group(
                        ax, values, bins, alpha_ci, shade_alpha, scatter_kw, label=label
                    )

        if multi_metric and not partition_keys:
            ax.legend()

        if config.xlabel is not None:
            ax.set_xlabel(config.xlabel)
        else:
            if not multi_metric:
                ax.set_xlabel(config.metric_labels.get(metric_names[0], metric_names[0]))
            else:
                ax.set_xlabel("Metric value")
        ax.set_ylabel(config.ylabel if config.ylabel is not None else "Frequency")

    def logical_error_rate(
        self,
        manager: SimulationDataManager,
        config: PlotConfig,
        per_round: int | None = None,
        return_num_shots: bool = False,
    ) -> float | tuple[float, int]:
        """Return the logical error rate for data matching *config*.

        Parameters
        ----------
        manager:
            Simulation data manager used for loading parquet data.
        config:
            Plot configuration; only selection filters are used.
        per_round:
            If an integer >= 1, return LER per round (LER / per_round).
        return_num_shots:
            When True, also return the number of shots (denominator).
        """
        metric_names = normalize_metric_names(config.metric_name)
        if len(metric_names) != 1:
            raise ValueError("logical_error_rate supports a single metric_name only")

        per_round = _normalize_per_round(per_round)
        lf = manager.query(config)
        return _logical_error_rate_from_lf(lf, per_round, return_num_shots)

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
        metric_names = normalize_metric_names(config.metric_name)
        if len(metric_names) != 1:
            raise ValueError("list_unique_values supports a single metric_name only")
        metric_name = metric_names[0]

        round_digits = _normalize_round_digits(config.plot_params.get("round_digits"))
        use_negative_gap = bool(config.plot_params.get("use_negative_gap", False))
        convert_db = bool(config.plot_params.get("convert_db", False))

        if use_negative_gap and metric_name not in _GAP_METRICS:
            raise ValueError(
                "use_negative_gap is only supported for gap metrics: "
                + ", ".join(sorted(_GAP_METRICS))
            )
        if convert_db and metric_name not in _GAP_METRICS:
            raise ValueError(
                "convert_db is only supported for gap metrics: "
                + ", ".join(sorted(_GAP_METRICS))
            )

        df = _collect_for_plot(lf, metric_name, [])
        values = self._extract_values(
            df, metric_name, round_digits, use_negative_gap, convert_db
        )
        if values.size == 0:
            return []

        return np.unique(values).tolist()

    def postselection_at_threshold(
        self,
        manager: SimulationDataManager,
        config: PlotConfig,
        threshold: float,
        direction: str | None = None,
        per_round: int | None = None,
        alpha: float = 0.05,
    ) -> PostselectionResult:
        """Compute post-selection statistics at a fixed metric threshold.

        Shots on the low-confidence side of *threshold* are aborted; the
        remaining shots constitute the post-selected sample.  The method
        returns the abort rate and the LER reduction rate so that the
        caller can evaluate the quality of post-selection at any specific
        operating point without sweeping an entire curve.

        Parameters
        ----------
        manager:
            Simulation data manager used to load parquet data.
        config:
            Plot configuration.  ``filters``, ``decoder_names``, and
            ``batch_indices`` are used for data selection exactly as in
            the existing plot methods; ``metric_name`` must be a single
            metric string.
        threshold:
            The threshold value applied to the metric column.
        direction:
            Which side of the threshold to **keep**:

            * ``"high"`` — keep shots with ``metric >= threshold``
              (abort below; appropriate for gap-style metrics).
            * ``"low"``  — keep shots with ``metric <= threshold``
              (abort above; appropriate for ``cluster_llr``).

            When *None*, the direction is inferred from the metric name
            using the same rules as :class:`PostSelectionPlotter`.
        per_round:
            When an integer >= 1, all LER values are divided by this
            number (LER per round).
        alpha:
            Wilson CI significance level for *post_ler* (default 0.05
            → 95 % CI).

        Returns
        -------
        PostselectionResult
            Dataclass containing ``abort_rate``, ``post_ler``,
            ``original_ler``, ``reduction_rate``, ``n_accepted``,
            ``n_total``, ``ci_low``, ``ci_high``.

        Raises
        ------
        ValueError
            If ``config.metric_name`` contains more than one metric, or
            if *direction* is invalid, or if the metric is a boolean
            metric that does not support threshold-based post-selection.
        """
        from analysis.src.postselect import (
            BOOLEAN_METRICS as _BOOLEAN_METRICS,
            _infer_direction,
        )

        metric_names = normalize_metric_names(config.metric_name)
        if len(metric_names) != 1:
            raise ValueError(
                "postselection_at_threshold supports a single metric_name only"
            )
        metric_name = metric_names[0]

        if direction is None:
            if metric_name in _BOOLEAN_METRICS:
                raise ValueError(
                    f"metric '{metric_name}' is a boolean metric; "
                    "threshold-based post-selection is not supported. "
                    "Use PostSelectionPlotter for boolean metrics."
                )
            direction = _infer_direction(metric_name)

        if direction not in {"high", "low"}:
            raise ValueError(
                f"direction must be 'high' or 'low', got {direction!r}"
            )

        per_round = _normalize_per_round(per_round)
        lf = manager.query(config)

        needed = [metric_name, "is_logical_error"]
        schema_names = set(lf.collect_schema().names())
        existing = [c for c in needed if c in schema_names]
        df = lf.select(existing).collect().drop_nulls(needed)

        values = df[metric_name].to_numpy().astype(float)
        is_error = df["is_logical_error"].to_numpy().astype(bool)
        n_total = int(values.size)

        _nan = float("nan")

        if n_total == 0:
            return PostselectionResult(
                abort_rate=_nan, post_ler=_nan, original_ler=_nan,
                reduction_rate=_nan, n_accepted=0, n_total=0,
                ci_low=_nan, ci_high=_nan,
            )

        original_ler = float(is_error.mean())
        if per_round is not None:
            original_ler /= per_round

        if direction == "high":
            mask = values >= threshold
        else:
            mask = values <= threshold

        n_accepted = int(mask.sum())
        abort_rate = 1.0 - n_accepted / n_total

        if n_accepted == 0:
            return PostselectionResult(
                abort_rate=abort_rate, post_ler=_nan,
                original_ler=original_ler, reduction_rate=_nan,
                n_accepted=0, n_total=n_total, ci_low=_nan, ci_high=_nan,
            )

        k_err = int(is_error[mask].sum())
        post_ler = k_err / n_accepted
        if per_round is not None:
            post_ler /= per_round

        ci_low_v, ci_high_v = wilson_ci(k_err, n_accepted, alpha=alpha)
        ci_low_v = float(ci_low_v)
        ci_high_v = float(ci_high_v)
        if per_round is not None:
            ci_low_v /= per_round
            ci_high_v /= per_round

        reduction_rate = post_ler / original_ler if original_ler > 0 else _nan

        return PostselectionResult(
            abort_rate=abort_rate,
            post_ler=post_ler,
            original_ler=original_ler,
            reduction_rate=reduction_rate,
            n_accepted=n_accepted,
            n_total=n_total,
            ci_low=ci_low_v,
            ci_high=ci_high_v,
        )

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
        convert_db: bool = False,
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

        if convert_db:
            values = _DB_FACTOR * values

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
        metric_names = normalize_metric_names(config.metric_name)
        multi_metric = len(metric_names) > 1
        partition_keys = _effective_partition_keys(config)

        alpha_ci: float = config.plot_params.get("alpha_ci", 0.05)
        errorbar_kw = {k: v for k, v in config.plot_params.items()
                       if k not in _INTERNAL_PLOT_PARAMS}

        style_map = _metric_style_map(metric_names, ax) if multi_metric else {}

        for metric_name in metric_names:
            df = _collect_for_plot(lf, metric_name, partition_keys)
            local_kw = dict(errorbar_kw)

            if multi_metric:
                local_kw.pop("color", None)
                local_kw.pop("c", None)
                local_kw.pop("marker", None)
                local_kw.pop("fmt", None)
                style = style_map[metric_name]
                local_kw["color"] = style["color"]
                fmt = style["marker"]
            else:
                fmt = local_kw.pop("fmt", "o")

            col = df[metric_name]
            accept_series = col if col.dtype == pl.Boolean else (col > 0.5)

            if not partition_keys:
                n = len(accept_series)
                k = int(accept_series.sum())
                rate = k / n if n > 0 else 0.0
                ci_low, ci_high = (wilson_ci(k, n, alpha=alpha_ci) if n > 0
                                   else (0.0, 0.0))
                label = metric_name if multi_metric else None
                ax.errorbar(
                    [0], [rate],
                    yerr=[[rate - float(ci_low)], [float(ci_high) - rate]],
                    fmt=fmt, capsize=5, label=label, **local_kw,
                )
                ax.set_xticks([0])
                ax.set_xticklabels(["all"])
                ax.set_ylabel("Accept rate")
                ax.set_ylim(0, 1)
                continue

            partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                partition_keys, as_dict=True
            )

            labels: list[str] = []
            rates: list[float] = []
            ci_lows: list[float] = []
            ci_highs: list[float] = []

            for key_vals in sorted(partitions):
                part_col = partitions[key_vals][metric_name]
                part_accept = part_col if part_col.dtype == pl.Boolean else (part_col > 0.5)
                n = len(part_accept)
                k = int(part_accept.sum())
                rate = k / n if n > 0 else 0.0
                ci_low, ci_high = (wilson_ci(k, n, alpha=alpha_ci) if n > 0
                                   else (0.0, 0.0))

                label = _make_label(partition_keys, key_vals)
                if multi_metric:
                    label = _metric_label(metric_name, label)

                labels.append(label)
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
                fmt=fmt, capsize=5, label=None, **local_kw,
            )
            ax.set_xticks(x_arr)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Accept rate")
            ax.set_ylim(0, 1)

        if multi_metric and not partition_keys:
            ax.legend()

    def logical_error_rate(
        self,
        manager: SimulationDataManager,
        config: PlotConfig,
        per_round: int | None = None,
        return_num_shots: bool = False,
    ) -> float | tuple[float, int]:
        """Return the logical error rate for data matching *config*.

        Parameters
        ----------
        manager:
            Simulation data manager used for loading parquet data.
        config:
            Plot configuration; only selection filters are used.
        per_round:
            If an integer >= 1, return LER per round (LER / per_round).
        return_num_shots:
            When True, also return the number of shots (denominator).
        """
        metric_names = normalize_metric_names(config.metric_name)
        if len(metric_names) != 1:
            raise ValueError("logical_error_rate supports a single metric_name only")

        per_round = _normalize_per_round(per_round)
        lf = manager.query(config)
        return _logical_error_rate_from_lf(lf, per_round, return_num_shots)


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
    * ``"forced_gap_ml"``

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
        metric_names = normalize_metric_names(config.metric_name)
        multi_metric = len(metric_names) > 1
        style_map = _metric_style_map(metric_names, ax) if multi_metric else {}

        for metric_name in metric_names:
            _validate_conditional_metric(metric_name)
            metric_config = replace(config, metric_name=metric_name)
            df = self._collect(lf, metric_config)
            display_name = config.metric_labels.get(metric_name, metric_name)

            style = style_map.get(metric_name)
            if not metric_config.group_by:
                label = display_name if multi_metric else None
                self._plot_ler_group(
                    ax, df, metric_config, label=label, style=style
                )
            else:
                partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                    metric_config.group_by, as_dict=True
                )
                for key_vals in sorted(partitions):
                    label = _make_label(metric_config.group_by, key_vals)
                    if multi_metric:
                        label = _metric_label(display_name, label)
                    self._plot_ler_group(
                        ax, partitions[key_vals], metric_config, label=label, style=style
                    )

        if config.group_by or multi_metric:
            ax.legend()

        if config.xlabel is not None:
            ax.set_xlabel(config.xlabel)
        else:
            if not multi_metric:
                ax.set_xlabel(config.metric_labels.get(metric_names[0], metric_names[0]))
            else:
                ax.set_xlabel("Metric value")

        if config.ylabel is not None:
            ax.set_ylabel(config.ylabel)
        else:
            if not multi_metric:
                display0 = config.metric_labels.get(metric_names[0], metric_names[0])
                ax.set_ylabel(f"P(logical error | {display0})")
            else:
                ax.set_ylabel("P(logical error | metric)")

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

        metric_names = normalize_metric_names(config.metric_name)
        multi_metric = len(metric_names) > 1
        style_map = _metric_style_map(metric_names, ax) if multi_metric else {}

        for metric_name in metric_names:
            _validate_conditional_metric(metric_name)
            metric_config = replace(config, metric_name=metric_name)
            df = self._collect(lf, metric_config)
            display_name = config.metric_labels.get(metric_name, metric_name)

            style = style_map.get(metric_name)
            if not metric_config.group_by:
                label = display_name if multi_metric else None
                self._plot_fitting_group(
                    ax, df, metric_config, label=label, style=style
                )
            else:
                partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                    metric_config.group_by, as_dict=True
                )
                for key_vals in sorted(partitions):
                    label = _make_label(metric_config.group_by, key_vals)
                    if multi_metric:
                        label = _metric_label(display_name, label)
                    self._plot_fitting_group(
                        ax, partitions[key_vals], metric_config, label=label, style=style
                    )

        if config.group_by or multi_metric:
            ax.legend()

        if config.xlabel is not None:
            ax.set_xlabel(config.xlabel)
        else:
            if not multi_metric:
                ax.set_xlabel(config.metric_labels.get(metric_names[0], metric_names[0]))
            else:
                ax.set_xlabel("Metric value")

        ax.set_ylabel(
            config.ylabel if config.ylabel is not None
            else r"$\log\!\left(\dfrac{1 - y}{y}\right)$"
        )

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
        if config.convert_db:
            x = _DB_FACTOR * x
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
        style: dict[str, Any] | None = None,
    ) -> None:
        x, y = self._metric_and_error(df, config)
        if x.size == 0:
            return

        bstats = self._ler_stats(x, y, config)
        valid = ~np.isnan(bstats.proportions)

        scatter_kw = {"s": 30}
        if style:
            scatter_kw.update(style)
        sc = ax.scatter(
            bstats.centers[valid], bstats.proportions[valid],
            label=label, **scatter_kw,
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
        style: dict[str, Any] | None = None,
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
        scatter_kw = {"s": 30}
        if style:
            scatter_kw.update(style)
        sc = ax.scatter(g, z, label=data_label, **scatter_kw)
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
