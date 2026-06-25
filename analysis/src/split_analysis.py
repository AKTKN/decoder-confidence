"""Split-parameter analysis helpers for decoder statistics.

This module focuses on data generated with ``get_detail_stat=True``.  The
``get_detail_stat`` flag is treated as a storage detail, while split-related
parameters such as ``random_split``, ``n_splits``, and ``split_seed`` are exposed
as experiment parameters that can be compared in notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis.src.data_manager import (
    _extract_batch_idx,
    _infer_literal,
    _is_circuit_params_dir,
    _matches_all_filters,
    _parse_kv,
    _strip_quotes,
)


SPLIT_METRICS: frozenset[str] = frozenset({"linearize_logicalgap", "forced_gap_ml"})
SPLIT_KEYS: tuple[str, ...] = ("random_split", "randon_split", "n_splits")


@dataclass(frozen=True)
class DecoderStatExperimentSpec:
    """Describe one decoder-stat dataset to load and compare.

    Parameters
    ----------
    metric_name:
        Metric whose decoder-stat directory should be loaded.
    decoder_names:
        Optional decoder-name whitelist. ``None`` loads all matching decoders.
    filters:
        Circuit-parameter filters, for example ``{"d": 6, "p": 0.003}``.
    random_split, n_splits:
        Split configuration for ``linearize_logicalgap`` and ``forced_gap_ml``.
        They must be specified together. ``None`` selects the ordinary
        non-split metric directory.
    split_seed:
        Optional split seed filter.
    label:
        Human-readable series label used in tables and plots.
    batch_indices:
        Optional batch whitelist. ``None`` loads all decoder-stat batches.
    decoder_filters:
        Additional decoder-directory filters for advanced comparisons.
    """

    metric_name: str
    decoder_names: list[str] | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    random_split: bool | None = None
    n_splits: int | None = None
    split_seed: int | None = None
    label: str | None = None
    batch_indices: list[int] | None = None
    decoder_filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metric_name in SPLIT_METRICS:
            split_values = (self.random_split, self.n_splits)
            if any(value is not None for value in split_values) and any(
                value is None for value in split_values
            ):
                raise ValueError("random_split and n_splits must be specified together")
        elif self.random_split is not None or self.n_splits is not None:
            raise ValueError(
                "random_split/n_splits options are only supported for "
                "linearize_logicalgap and forced_gap_ml"
            )

    @property
    def display_label(self) -> str:
        """Return a stable label for tables and legends."""
        if self.label:
            return self.label
        parts = [self.metric_name]
        if self.decoder_names:
            parts.append("/".join(self.decoder_names))
        if self.random_split is not None:
            parts.append(f"random_split={self.random_split}")
            parts.append(f"n_splits={self.n_splits}")
        if self.split_seed is not None:
            parts.append(f"split_seed={self.split_seed}")
        return ", ".join(parts)

    def decoder_directory_filters(self) -> tuple[dict[str, Any], list[str]]:
        """Return include and exclude filters for decoder-result directories."""
        include = dict(self.decoder_filters)
        exclude: list[str] = []

        if self.metric_name not in SPLIT_METRICS:
            return include, exclude

        if self.random_split is None:
            exclude.extend(SPLIT_KEYS)
            return include, exclude

        include["n_splits"] = int(self.n_splits)
        if self.split_seed is not None:
            include["split_seed"] = int(self.split_seed)
        include["random_split"] = bool(self.random_split)
        return include, exclude


class DecoderStatDataLoader:
    """Load ``decoder_stat_batch=*.parquet`` files for split analyses."""

    def __init__(self, result_dir_root: Path) -> None:
        self.result_dir_root = Path(result_dir_root)
        if not self.result_dir_root.is_dir():
            raise FileNotFoundError(f"result_dir_root not found: {self.result_dir_root}")

    def load(self, specs: Sequence[DecoderStatExperimentSpec]) -> pl.LazyFrame:
        """Return a lazy table containing all decoder-stat rows for *specs*."""
        frames: list[pl.LazyFrame] = []
        for spec in specs:
            frames.extend(self._frames_for_spec(spec))
        if not frames:
            labels = ", ".join(spec.display_label for spec in specs)
            raise FileNotFoundError(
                f"No decoder_stat parquet files found for specs: {labels}"
            )
        return pl.concat(frames, how="diagonal")

    def _frames_for_spec(self, spec: DecoderStatExperimentSpec) -> list[pl.LazyFrame]:
        frames: list[pl.LazyFrame] = []
        for circuit_dir in sorted(self.result_dir_root.iterdir()):
            if not circuit_dir.is_dir() or not _is_circuit_params_dir(circuit_dir.name):
                continue
            circuit_params = _parse_kv(circuit_dir.name)
            if not _matches_all_filters(circuit_params, spec.filters):
                continue

            decoding_dir = circuit_dir / "decoding_result"
            if not decoding_dir.is_dir():
                continue

            for decoder_dir in sorted(decoding_dir.iterdir()):
                if not decoder_dir.is_dir():
                    continue
                decoder_params = _parse_kv(decoder_dir.name)
                if _strip_quotes(decoder_params.get("metric", "")) != spec.metric_name:
                    continue
                decoder = _strip_quotes(decoder_params.get("decoder", ""))
                if spec.decoder_names is not None and decoder not in spec.decoder_names:
                    continue
                include, exclude = spec.decoder_directory_filters()
                if include and not _matches_all_filters(decoder_params, include):
                    continue
                if any(key in decoder_params for key in exclude):
                    continue

                frames.extend(
                    self._frames_for_decoder_dir(
                        decoder_dir=decoder_dir,
                        params={**circuit_params, **decoder_params},
                        spec=spec,
                    )
                )
        return frames

    def _frames_for_decoder_dir(
        self,
        decoder_dir: Path,
        params: dict[str, str],
        spec: DecoderStatExperimentSpec,
    ) -> list[pl.LazyFrame]:
        frames: list[pl.LazyFrame] = []
        for stat_file in sorted(decoder_dir.glob("decoder_stat_batch=*.parquet")):
            batch_idx = _extract_batch_idx(stat_file.name)
            if batch_idx is None:
                continue
            if spec.batch_indices is not None and batch_idx not in spec.batch_indices:
                continue

            literal_exprs = [_infer_literal(val).alias(key) for key, val in params.items()]
            literal_exprs.extend(
                [
                    pl.lit(batch_idx, dtype=pl.Int64).alias("batch"),
                    pl.lit(spec.display_label).alias("experiment"),
                ]
            )
            frames.append(pl.scan_parquet(stat_file).with_columns(literal_exprs))
        return frames


def load_decoder_stats(
    result_dir_root: Path,
    specs: Sequence[DecoderStatExperimentSpec],
) -> pl.LazyFrame:
    """Convenience wrapper around :class:`DecoderStatDataLoader`."""
    return DecoderStatDataLoader(result_dir_root).load(specs)


def decoder_stat_columns(lf: pl.LazyFrame) -> list[str]:
    """Return numeric decoder-stat columns, excluding identifiers and metadata."""
    schema = lf.collect_schema()
    excluded = {
        "shot_id",
        "batch",
        "experiment",
        "decoder",
        "metric",
        "code",
        "noisemodel",
        "d",
        "rounds",
        "p",
        "b",
        "alpha",
        "cluster_llr_alpha",
        "n_splits",
        "split_seed",
        "num_decoding_rounds",
        "use_both",
        "xyz",
        "ibm_reproduce",
        "random_split",
        "randon_split",
        "get_detail_stat",
    }
    numeric = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    return [
        name
        for name, dtype in schema.items()
        if name not in excluded and dtype in numeric
    ]


def summarize_decoder_stats(
    lf: pl.LazyFrame,
    columns: Sequence[str] | None = None,
    group_by: Sequence[str] = ("experiment",),
) -> pl.DataFrame:
    """Compute basic statistics for decoder-stat columns.

    The returned table is long-form: one row per ``group_by`` combination and
    statistic column.  This format is convenient for notebook display and
    downstream filtering.
    """
    selected = list(columns) if columns is not None else decoder_stat_columns(lf)
    if not selected:
        raise ValueError("No decoder-stat columns were selected")

    rows: list[pl.LazyFrame] = []
    keys = list(group_by)
    for col in selected:
        expr = pl.col(col).cast(pl.Float64)
        rows.append(
            lf.group_by(keys)
            .agg(
                [
                    pl.len().alias("rows"),
                    pl.col(col).null_count().alias("null_count"),
                    expr.drop_nans().drop_nulls().count().alias("finite_count"),
                    expr.mean().alias("mean"),
                    expr.std().alias("std"),
                    expr.min().alias("min"),
                    expr.quantile(0.25).alias("q25"),
                    expr.median().alias("median"),
                    expr.quantile(0.75).alias("q75"),
                    expr.max().alias("max"),
                ]
            )
            .with_columns(pl.lit(col).alias("stat_column"))
        )

    return (
        pl.concat(rows, how="diagonal")
        .select(keys + ["stat_column", "rows", "null_count", "finite_count",
                        "mean", "std", "min", "q25", "median", "q75", "max"])
        .sort(keys + ["stat_column"])
        .collect()
    )


def value_frequency_table(
    lf: pl.LazyFrame,
    columns: Sequence[str],
    group_by: Sequence[str] = ("experiment",),
    round_digits: int | None = None,
) -> pl.DataFrame:
    """Return value-frequency rows for the selected decoder-stat columns."""
    if not columns:
        raise ValueError("columns must be non-empty")

    keys = list(group_by)
    rows: list[pl.LazyFrame] = []
    for col in columns:
        value = pl.col(col).cast(pl.Float64)
        if round_digits is not None:
            value = value.round(round_digits)
        rows.append(
            lf.select(keys + [value.alias("value")])
            .drop_nulls(["value"])
            .filter(pl.col("value").is_finite())
            .group_by(keys + ["value"])
            .agg(pl.len().alias("frequency"))
            .with_columns(pl.lit(col).alias("stat_column"))
        )
    return (
        pl.concat(rows, how="diagonal")
        .sort(keys + ["stat_column", "value"])
        .collect()
    )


def plot_decoder_stat_frequency(
    lf: pl.LazyFrame,
    columns: Sequence[str],
    ax: plt.Axes,
    *,
    group_by: Sequence[str] = ("experiment",),
    round_digits: int | None = None,
    label_template: str = "{experiment} | {stat_column}",
    scatter_kw: dict[str, Any] | None = None,
) -> plt.Axes:
    """Scatter plot of decoder-stat value frequencies.

    The x-axis is the selected statistic value and the y-axis is its frequency.
    Multiple statistic columns and multiple experiments are overlaid by forming
    one series for each ``group_by`` + ``stat_column`` combination.
    """
    df = value_frequency_table(
        lf,
        columns=columns,
        group_by=group_by,
        round_digits=round_digits,
    )
    scatter_kw = {"s": 18, "alpha": 0.75, **(scatter_kw or {})}
    keys = list(group_by) + ["stat_column"]
    partitions = df.partition_by(keys, as_dict=True)
    for key_vals, part in partitions.items():
        key_tuple = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        label_values = dict(zip(keys, key_tuple))
        label = label_template.format(**label_values)
        ax.scatter(
            part["value"].to_numpy(),
            part["frequency"].to_numpy(),
            label=label,
            **scatter_kw,
        )
    return ax


def make_split_postselect_specs(
    *,
    metric_name: str,
    decoder_names: list[str] | None,
    filters: dict[str, Any],
    split_configs: Iterable[tuple[bool | None, int | None, int | None, str]],
    batch_indices: list[int] | None = None,
    get_detail_stat: bool | None = True,
) -> list[Any]:
    """Build ``PostSelectSpec`` objects for ordinary and split comparisons.

    Each split config is ``(random_split, n_splits, split_seed, label)``.
    The function imports ``PostSelectSpec`` lazily to avoid a module-level cycle.
    """
    from analysis.src.postselect import PostSelectSpec

    return [
        PostSelectSpec(
            metric_name=metric_name,
            decoder_names=decoder_names,
            filters=dict(filters),
            get_detail_stat=get_detail_stat,
            random_split=random_split,
            n_splits=n_splits,
            split_seed=split_seed,
            label_prefix=label,
            batch_indices=batch_indices,
        )
        for random_split, n_splits, split_seed, label in split_configs
    ]


def finite_column_values(df: pl.DataFrame, column: str) -> np.ndarray:
    """Return finite values from a collected decoder-stat table column."""
    return (
        df.select(pl.col(column).cast(pl.Float64).alias(column))
        .drop_nulls(column)
        .filter(pl.col(column).is_finite())
        .get_column(column)
        .to_numpy()
    )
