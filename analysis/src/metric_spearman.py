"""Spearman rank-correlation analysis for numeric simulation metrics.

Given a set of metrics (each optionally restricted to specific decoders),
all are loaded, joined on shot_id, and all pairwise Spearman correlations are
computed and visualised as an annotated heatmap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from analysis.src.config import PlotConfig
from analysis.src.data_manager import SimulationDataManager

# shot_id is reset to 0 for each batch (.b8 file), so "batch" must be included
# to uniquely identify a shot across batches.
_CIRCUIT_PARAM_KEYS: list[str] = ["batch", "code", "d", "p", "rounds", "noisemodel", "xyz"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpearmanConfig:
    """Configuration for pairwise Spearman rank-correlation analysis.

    All correlations are computed in terms of **abort priority rank**:
    the shot that should be aborted first receives rank 1.

    * For gap-like metrics (logical_gap, forced_gap_ml, etc.): small value
      → high abort priority.  Values are used as-is.
    * For ``cluster_llr``: large value → high abort priority.  List it in
      ``negate_metrics`` so its values are negated before ranking, aligning
      the direction with the gap metrics.

    Parameters
    ----------
    metric_specs :
        List of ``(metric_name, decoder_names)`` pairs. ``decoder_names`` may
        be ``None`` to include all decoders for that metric.
    filters :
        Experiment parameter filters applied to every metric.
    batch_indices :
        Restrict to specific batch indices. ``None`` loads all batches.
    metric_labels :
        Human-readable axis labels keyed by metric name.
    negate_metrics :
        Metric names whose values are negated before computing rank
        correlations.  Use this for metrics where *larger* values indicate
        higher abort priority (e.g. ``"cluster_llr"``).
        The negation is reflected in the axis label with a leading ``-``.
    """

    metric_specs: list[tuple[str, list[str] | None]]
    filters: dict[str, Any] = field(default_factory=dict)
    batch_indices: list[int] | None = None
    metric_labels: dict[str, str] = field(default_factory=dict)
    negate_metrics: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_metric(
    manager: SimulationDataManager,
    metric_name: str,
    decoder_names: list[str] | None,
    filters: dict[str, Any],
    batch_indices: list[int] | None,
    join_keys: list[str],
) -> pl.DataFrame:
    """Load a single metric and return a DataFrame with join keys + metric column."""
    cfg = PlotConfig(
        metric_name=metric_name,
        decoder_names=decoder_names,
        filters=filters,
        batch_indices=batch_indices,
    )
    lf = manager.query(cfg)
    schema = set(lf.collect_schema().names())
    # Column name in the final parquet is just the metric name (no "metric_" prefix).
    col = metric_name
    select = [k for k in join_keys if k in schema] + ([col] if col in schema else [])
    return lf.select(select).collect()


def _join_all_metrics(
    manager: SimulationDataManager,
    config: SpearmanConfig,
) -> tuple[pl.DataFrame, list[str]]:
    """Load all metrics and inner-join them on shot_id + circuit params.

    Returns
    -------
    joined :
        DataFrame containing all metric columns.
    metric_cols :
        Ordered list of ``metric_<name>`` column names present in *joined*.
    """
    if not config.metric_specs:
        raise ValueError("metric_specs must be non-empty")

    # Determine join keys from the first metric's schema.
    first_name, first_decoders = config.metric_specs[0]
    first_cfg = PlotConfig(
        metric_name=first_name,
        decoder_names=first_decoders,
        filters=config.filters,
        batch_indices=config.batch_indices,
    )
    first_schema = set(manager.query(first_cfg).collect_schema().names())
    join_keys = ["shot_id"] + [
        k for k in _CIRCUIT_PARAM_KEYS if k in first_schema
    ]

    dfs: list[pl.DataFrame] = []
    for metric_name, decoder_names in config.metric_specs:
        df = _load_metric(
            manager, metric_name, decoder_names,
            config.filters, config.batch_indices, join_keys,
        )
        dfs.append(df)

    # Sequential inner-join accumulation.
    actual_keys = [k for k in join_keys if k in dfs[0].columns]
    joined = dfs[0]
    for df in dfs[1:]:
        keys_here = [k for k in actual_keys if k in df.columns]
        joined = joined.join(df, on=keys_here, how="inner")

    metric_cols = [
        name
        for name, _ in config.metric_specs
        if name in joined.columns
    ]
    return joined, metric_cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_spearman_matrix(
    manager: SimulationDataManager,
    config: SpearmanConfig,
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise Spearman rank correlations.

    Parameters
    ----------
    manager :
        Simulation data manager.
    config :
        Spearman configuration.

    Returns
    -------
    corr_matrix :
        Square (n_metrics × n_metrics) array of Spearman ρ values.
    labels :
        Ordered list of display labels corresponding to the matrix rows/columns.
    """
    joined, metric_cols = _join_all_metrics(manager, config)

    negate = set(config.negate_metrics)
    n = len(metric_cols)
    corr_matrix = np.full((n, n), np.nan)

    def _get_array(col: str) -> np.ndarray:
        arr = joined[col].to_numpy().astype(float)
        return -arr if col in negate else arr

    # For each pair, align on finite values by re-selecting from the joined df.
    for i in range(n):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sub = joined.select([metric_cols[i], metric_cols[j]]).drop_nulls()
            if sub.height < 3:
                continue
            x = sub[metric_cols[i]].to_numpy().astype(float)
            y = sub[metric_cols[j]].to_numpy().astype(float)
            if metric_cols[i] in negate:
                x = -x
            if metric_cols[j] in negate:
                y = -y
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            rho, _ = stats.spearmanr(x[mask], y[mask])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # Labels: prepend "-" for negated metrics to make the direction explicit.
    def _label(col: str) -> str:
        base = config.metric_labels.get(col, col)
        return f"-{base}" if col in negate else base

    labels = [_label(col) for col in metric_cols]
    return corr_matrix, labels


def plot_spearman_heatmap(
    manager: SimulationDataManager,
    config: SpearmanConfig,
    *,
    ax: plt.Axes | None = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot_fmt: str = ".3f",
    annot_fontsize: int = 10,
    print_matrix: bool = True,
) -> tuple[plt.Axes, np.ndarray, list[str]]:
    """Plot an annotated Spearman rank-correlation heatmap.

    Parameters
    ----------
    manager :
        Simulation data manager.
    config :
        Spearman configuration.
    ax :
        Axes to draw on. A new figure is created when ``None``.
    cmap :
        Matplotlib colormap name (default ``"RdBu_r"``).
    vmin, vmax :
        Color scale limits.
    annot_fmt :
        Format string for the numeric annotations in each cell.
    annot_fontsize :
        Font size for the cell annotations.
    print_matrix :
        Print the correlation matrix to stdout.

    Returns
    -------
    ax :
        The axes containing the heatmap.
    corr_matrix :
        (n × n) array of Spearman ρ values.
    labels :
        Display labels for the rows/columns.
    """
    corr_matrix, labels = compute_spearman_matrix(manager, config)

    if print_matrix:
        n = len(labels)
        header = "  ".join(f"{lb:>12s}" for lb in labels)
        print(f"{'':>12s}  {header}")
        for i, row_label in enumerate(labels):
            row_str = "  ".join(
                f"{corr_matrix[i, j]:>12{annot_fmt[1:]}}"
                if not np.isnan(corr_matrix[i, j])
                else f"{'nan':>12s}"
                for j in range(n)
            )
            print(f"{row_label:>12s}  {row_str}")

    if ax is None:
        n = len(labels)
        figsize = max(4, n * 1.2)
        _, ax = plt.subplots(figsize=(figsize, figsize * 0.85))

    im = ax.imshow(corr_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            text = f"{val:{annot_fmt}}" if not np.isnan(val) else "nan"
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=annot_fontsize, color=text_color)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Metric")

    return ax, corr_matrix, labels
