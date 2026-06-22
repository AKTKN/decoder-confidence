from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import polars as pl

from analysis.src.data_manager import (
    _extract_batch_idx,
    _infer_literal,
    _is_circuit_params_dir,
    _matches_all_filters,
    _parse_kv,
    _strip_quotes,
)

FORCING_DEGRADATION_METRIC = "forcing_degradation_test"
ForcingStage = Literal["baseline", "forced"]


@dataclass
class ForcingDegradationConfig:
    """Configuration for loading forcing-degradation simulation results.

    Parameters
    ----------
    filters:
        Key/value conditions applied to circuit-parameter directory names,
        such as ``{"code": "surface_code_Z", "d": 5, "p": 0.001}``.
        Values are compared with the same coercion rules as
        :class:`analysis.src.data_manager.SimulationDataManager`.
    decoder_names:
        Optional decoder directory names to include, for example
        ``["BP-LSD"]`` or ``["RELAY-BP"]``.  ``None`` includes every
        forcing-degradation result directory found under matching circuits.
    batch_indices:
        Optional one-based sampled-data batch indices.  ``None`` loads all
        available ``metric=forcing_degradation_test_batch=N.parquet`` files.
    """

    filters: dict[str, Any] = field(default_factory=dict)
    decoder_names: list[str] | None = None
    batch_indices: list[int] | None = None


@dataclass
class ForcingPlotOptions:
    """Plot controls for forcing-degradation metric-frequency figures.

    Parameters
    ----------
    figsize:
        Figure size passed to :func:`matplotlib.pyplot.subplots` when ``ax`` is
        not supplied.
    xlabel, ylabel:
        Axis labels.  When ``xlabel`` is ``None``, the plotted metric column
        name is used.
    xlim, ylim:
        Optional axis limits.
    marker_size:
        Marker area passed to ``Axes.scatter``.
    alpha:
        Marker transparency.
    show_legend:
        Whether to draw a legend.
    legend_loc:
        Matplotlib legend location.
    font_size:
        Optional rcParam font size applied while creating the figure.
    output_filename:
        Optional output path.  When set, the figure is saved before returning.
    output_format:
        Optional format passed to ``Figure.savefig``.
    dpi:
        Output DPI for saved figures.
    use_latex:
        Whether to enable ``text.usetex`` while creating the figure.
    distinguish_logicalerror:
        If true, split frequencies by baseline logical-error status and draw
        logical-error shots as red crosses and non-logical-error shots as blue
        circles.
    """

    figsize: tuple[float, float] = (6.0, 4.0)
    xlabel: str | None = None
    ylabel: str = "Frequency"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    marker_size: float = 24.0
    alpha: float = 0.8
    show_legend: bool = True
    legend_loc: str = "best"
    font_size: float | None = None
    output_filename: str | Path | None = None
    output_format: str | None = None
    dpi: int = 300
    use_latex: bool = True
    distinguish_logicalerror: bool = False


def load_forcing_degradation_lazy(
    result_dir_root: Path,
    config: ForcingDegradationConfig,
) -> pl.LazyFrame:
    """Load forcing-degradation shot tables as a Polars LazyFrame.

    Parameters
    ----------
    result_dir_root:
        Root directory containing circuit-parameter subdirectories.
    config:
        Directory filters, decoder filters, and batch-index filters.

    Returns
    -------
    polars.LazyFrame
        Concatenated shot table.  Required shot columns include ``shot_id``,
        ``logicalerror``, ``baseline_weight``, and ``forced_weight``.  Depending
        on decoder, either ``baseline_cluster_llr``/``forced_cluster_llr`` or
        ``baseline_iteration``/``forced_iteration`` are present.  Path-encoded
        circuit and decoder parameters are appended as literal columns, along
        with ``batch``.

    Raises
    ------
    FileNotFoundError
        If no matching result files are found.
    """
    root = Path(result_dir_root)
    if not root.is_dir():
        raise FileNotFoundError(f"result_dir_root not found: {root}")

    frames: list[pl.LazyFrame] = []
    for circuit_dir in sorted(root.iterdir()):
        if not circuit_dir.is_dir() or not _is_circuit_params_dir(circuit_dir.name):
            continue
        circuit_params = _parse_kv(circuit_dir.name)
        if not _matches_all_filters(circuit_params, config.filters):
            continue

        decoding_dir = circuit_dir / "decoding_result"
        if not decoding_dir.is_dir():
            continue

        for result_dir in sorted(decoding_dir.iterdir()):
            if not result_dir.is_dir():
                continue
            result_params = _parse_kv(result_dir.name)
            if _strip_quotes(result_params.get("metric", "")) != FORCING_DEGRADATION_METRIC:
                continue
            decoder = _strip_quotes(result_params.get("decoder", ""))
            if config.decoder_names is not None and decoder not in config.decoder_names:
                continue

            all_params = {**circuit_params, **result_params}
            for metric_file in sorted(
                result_dir.glob(f"metric={FORCING_DEGRADATION_METRIC}_batch=*.parquet")
            ):
                batch_idx = _extract_batch_idx(metric_file.name)
                if batch_idx is None:
                    continue
                if config.batch_indices is not None and batch_idx not in config.batch_indices:
                    continue
                literal_exprs = [
                    _infer_literal(value).alias(key)
                    for key, value in all_params.items()
                ]
                literal_exprs.append(pl.lit(batch_idx, dtype=pl.Int64).alias("batch"))
                frames.append(pl.scan_parquet(metric_file).with_columns(literal_exprs))

    if not frames:
        raise FileNotFoundError(
            "No forcing_degradation_test parquet data found "
            f"with filters={config.filters} under {root}"
        )
    return pl.concat(frames, how="diagonal")


def forcing_metric_columns(schema: Mapping[str, Any] | Sequence[str]) -> list[str]:
    """Return stage-prefixed decoder-specific metric columns from a schema."""
    names = list(schema.keys()) if hasattr(schema, "keys") else list(schema)
    baseline = [
        name
        for name in names
        if name.startswith("baseline_")
        and name not in {"baseline_weight"}
        and f"forced_{name.removeprefix('baseline_')}" in names
    ]
    forced = [f"forced_{name.removeprefix('baseline_')}" for name in baseline]
    return [item for pair in zip(baseline, forced) for item in pair]


def forcing_value_columns(schema: Mapping[str, Any] | Sequence[str]) -> list[str]:
    """Return forcing-degradation numeric value columns worth summarizing.

    The returned order is stable and notebook-friendly: solution weights first,
    then decoder-specific baseline/forced metrics.
    """
    names = list(schema.keys()) if hasattr(schema, "keys") else list(schema)
    columns = [name for name in ("baseline_weight", "forced_weight") if name in names]
    columns.extend(forcing_metric_columns(names))
    return columns


def descriptive_statistics(
    data: pl.LazyFrame | pl.DataFrame,
    *,
    separate_logicalerror: bool = False,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> pl.DataFrame:
    """Compute descriptive statistics for forcing-degradation columns.

    Parameters
    ----------
    data:
        Lazy or eager forcing-degradation shot table.
    separate_logicalerror:
        If true, compute one statistics table for each value of baseline
        ``logicalerror``.  If false, use all shots together.
    quantiles:
        Quantiles to include for each numeric column.

    Returns
    -------
    polars.DataFrame
        Long-form table with columns ``logicalerror`` when requested,
        ``column``, ``count``, ``mean``, ``std``, ``min``, requested quantile
        columns named like ``q25`` and ``q50``, and ``max``.

    Raises
    ------
    ValueError
        If required forcing-degradation columns are missing.
    """
    df = data.collect() if isinstance(data, pl.LazyFrame) else data
    required = {"logicalerror", "baseline_weight", "forced_weight"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    value_cols = forcing_value_columns(df.columns)
    group_values: list[bool | None]
    if separate_logicalerror:
        group_values = sorted(df["logicalerror"].unique().to_list())
    else:
        group_values = [None]

    rows: list[dict[str, Any]] = []
    for logical_value in group_values:
        subset = (
            df.filter(pl.col("logicalerror") == logical_value)
            if logical_value is not None
            else df
        )
        for col in value_cols:
            series = subset[col].cast(pl.Float64)
            row: dict[str, Any] = {
                "column": col,
                "count": int(series.len()),
                "mean": float(series.mean()) if series.len() else None,
                "std": float(series.std()) if series.len() else None,
                "min": float(series.min()) if series.len() else None,
                "max": float(series.max()) if series.len() else None,
            }
            if separate_logicalerror:
                row["logicalerror"] = bool(logical_value)
            for q in quantiles:
                label = f"q{int(round(q * 100)):02d}"
                value = series.quantile(float(q)) if series.len() else None
                row[label] = float(value) if value is not None else None
            rows.append(row)

    order = ["logicalerror", "column"] if separate_logicalerror else ["column"]
    rest = [name for name in rows[0] if name not in order] if rows else []
    return pl.DataFrame(rows).select([*order, *rest]) if rows else pl.DataFrame()


def _metric_column_for_stage(df: pl.DataFrame, stage: ForcingStage) -> str:
    prefix = f"{stage}_"
    candidates = [
        name
        for name in df.columns
        if name.startswith(prefix) and name != f"{stage}_weight"
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {stage} decoder-specific metric column, "
            f"found {candidates}"
        )
    return candidates[0]


def plot_value_frequency(
    data: pl.LazyFrame | pl.DataFrame,
    *,
    column: str,
    options: ForcingPlotOptions | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot observation frequency for one forcing-degradation value column.

    Parameters
    ----------
    data:
        Lazy or eager forcing-degradation shot table.
    column:
        Numeric column to place on the horizontal axis, such as
        ``"baseline_weight"``, ``"forced_weight"``,
        ``"baseline_cluster_llr"``, or ``"forced_iteration"``.
    options:
        Plot appearance and output controls.
    ax:
        Optional Matplotlib axes.  If omitted, a new figure and axes are
        created using ``options.figsize``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes containing the plot.

    Raises
    ------
    ValueError
        If ``column`` is unavailable or ``logicalerror`` is missing when
        logical-error separation is requested.
    """
    options = options or ForcingPlotOptions()
    old_rc = {
        "text.usetex": plt.rcParams.get("text.usetex"),
        "font.size": plt.rcParams.get("font.size"),
    }
    plt.rcParams["text.usetex"] = options.use_latex
    if options.font_size is not None:
        plt.rcParams["font.size"] = options.font_size
    try:
        df = data.collect() if isinstance(data, pl.LazyFrame) else data
        if column not in df.columns:
            raise ValueError(f"Column {column!r} not found in forcing-degradation data")
        if options.distinguish_logicalerror and "logicalerror" not in df.columns:
            raise ValueError(
                "logicalerror column is required when distinguish_logicalerror=True"
            )

        group_cols = [column]
        if options.distinguish_logicalerror:
            group_cols.append("logicalerror")
        freq = (
            df.group_by(group_cols)
            .agg(pl.len().alias("frequency"))
            .sort(group_cols)
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=options.figsize)
        else:
            fig = ax.figure

        if options.distinguish_logicalerror:
            styles = [
                (False, "blue", "o", "non-logical error"),
                (True, "red", "x", "logical error"),
            ]
            for logical_value, color, marker, label in styles:
                part = freq.filter(pl.col("logicalerror") == logical_value)
                if part.is_empty():
                    continue
                ax.scatter(
                    part[column].to_numpy(),
                    part["frequency"].to_numpy(),
                    s=options.marker_size,
                    alpha=options.alpha,
                    c=color,
                    marker=marker,
                    label=label,
                )
        else:
            ax.scatter(
                freq[column].to_numpy(),
                freq["frequency"].to_numpy(),
                s=options.marker_size,
                alpha=options.alpha,
            )

        ax.set_xlabel(options.xlabel or column)
        ax.set_ylabel(options.ylabel)
        if options.xlim is not None:
            ax.set_xlim(*options.xlim)
        if options.ylim is not None:
            ax.set_ylim(*options.ylim)
        if options.show_legend and options.distinguish_logicalerror:
            ax.legend(loc=options.legend_loc)
        if options.output_filename is not None:
            fig.savefig(
                options.output_filename,
                format=options.output_format,
                dpi=options.dpi,
                bbox_inches="tight",
            )
        return fig, ax
    finally:
        plt.rcParams["text.usetex"] = old_rc["text.usetex"]
        if old_rc["font.size"] is not None:
            plt.rcParams["font.size"] = old_rc["font.size"]


def plot_metric_frequency(
    data: pl.LazyFrame | pl.DataFrame,
    *,
    stage: ForcingStage = "baseline",
    options: ForcingPlotOptions | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot decoder-specific metric frequency for one decoding stage.

    This is a convenience wrapper around :func:`plot_value_frequency` that
    automatically selects the decoder-specific metric column for ``stage``.
    """
    df = data.collect() if isinstance(data, pl.LazyFrame) else data
    metric_col = _metric_column_for_stage(df, stage)
    return plot_value_frequency(df, column=metric_col, options=options, ax=ax)
