"""Shot-level metric comparison: Concordance Correlation Coefficient and scatter plots.

Two metrics (potentially from different decoders) are joined on shot_id so that
each scatter point corresponds to the same physical shot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
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
    x_label :
        Override for the x-axis label. Defaults to ``x_metric``.
    y_label :
        Override for the y-axis label. Defaults to ``y_metric``.
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
    x_label: str | None = None
    y_label: str | None = None
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
    x_label = config.x_label or config.x_metric
    y_label = config.y_label or config.y_metric

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
        ax.legend()

    # y = x reference line spanning the data range (uses transformed values)
    x_all, y_all = _xy_from_part(df)
    if x_all.size > 0:
        all_vals = np.concatenate([x_all, y_all])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, zorder=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax, ccc_dict
