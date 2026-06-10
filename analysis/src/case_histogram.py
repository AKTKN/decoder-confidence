"""Histogram analysis for forced_gap_ml_case distributions.

The forced_gap_ml decoder, when run with ``get_all_failure_rate=True``,
produces a per-shot integer label stored as ``metric=forced_gap_ml_case``.
Each label classifies the outcome of the two-stage decoding relative to the
true observables for that shot:

    -1 : Normal success  — Stage 1 is correct and was adopted as the final answer.
         Does not fall into any of the failure-analysis cases below.
     0 : All K+1 fail    — All solutions (1 from Stage 1 + K from Stage 2)
         are logical errors.
     1 : Stage 2 rescue  — Stage 1 is a logical error; at least one Stage 2
         solution is correct, and that correct solution was adopted (minimum weight).
     2 : Missed rescue   — Stage 1 is a logical error; at least one Stage 2
         solution is correct, but it was NOT adopted (not minimum weight overall).
     3 : Stage 1 override — Stage 1 is NOT a logical error, but a Stage 2
         solution with strictly smaller weight was adopted instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis.src.config import PlotConfig
from analysis.src.data_manager import (
    SimulationDataManager,
    _extract_batch_idx,
    _infer_literal,
    _is_circuit_params_dir,
    _matches_all_filters,
    _parse_kv,
    _strip_quotes,
)

# Circuit-param columns used as additional join keys (mirrors metric_correlation.py).
# shot_id is reset per batch, so "batch" + circuit params are needed for uniqueness.
_CIRCUIT_PARAM_KEYS: list[str] = [
    "batch", "code", "d", "p", "rounds", "noisemodel", "xyz",
]

# Paired color palettes for color_by_logical_error mode.
# Index i: non-error color vs error color for group i.
_NON_ERROR_COLORS: list[str] = [
    "tab:blue", "tab:green", "tab:purple", "tab:cyan", "tab:olive",
]
_ERROR_COLORS: list[str] = [
    "tab:red", "tab:orange", "tab:brown", "tab:pink", "tab:gray",
]

# Short tick label and multi-line description for each case value
CASE_TICK_LABELS: dict[int, str] = {
    -1: "−1\n(success)",
    0: "0\n(all fail)",
    1: "1\n(rescued)",
    2: "2\n(missed)",
    3: "3\n(overridden)",
}

CASE_DESCRIPTIONS: dict[int, str] = {
    -1: "−1: Stage 1 correct, adopted (normal success)",
    0: " 0: All K+1 solutions are logical errors",
    1: " 1: Stage 1 error → correct Stage 2 solution adopted",
    2: " 2: Stage 1 error → correct Stage 2 solution existed but not adopted",
    3: " 3: Stage 1 correct but overridden by lower-weight Stage 2 solution",
}


def case_description_text(case_values: list[int]) -> str:
    """Return a multi-line description string for the given case values.

    Intended for display via ``ax.text(...)`` in the notebook, e.g. as a
    caption box below a :meth:`ForcedGapMLCaseAnalyzer.plot_case_histogram`
    figure.
    """
    return "\n".join(CASE_DESCRIPTIONS[v] for v in case_values)

@dataclass
class GapCaseHistogramConfig:
    """Configuration for case-distributed gap histograms.
    
    Parameters
    ----------
    decoder_names :
        Subset of base decoder names to include.
    filters :
        Circuit-parameter filters.
    batch_indices :
        Restrict loading to specific batch indices.
    gap_round_digits :
        Number of decimal places to round the gap value for binning.
        Defaults to 0 (groups by integer gap values).
    normalize_bars :
        If True, each bar shows the relative fraction of cases 0-3 (sums to 1.0).
        If False (default), shows raw counts.
    plot_negative_one :
        If True, overlays case -1 counts as a scatter plot.
    """
    decoder_names: Optional[List[str]] = None
    filters: dict[str, Any] = field(default_factory=dict)
    batch_indices: Optional[List[int]] = None
    gap_round_digits: int = 0
    normalize_bars: bool = False
    plot_negative_one: bool = False


@dataclass
class ForcedGapMLCaseConfig:
    """Configuration for :class:`ForcedGapMLCaseAnalyzer`.

    Parameters
    ----------
    decoder_names :
        Subset of base decoder names to include (matched against ``decoder=``
        in the directory name). ``None`` includes all decoders.
    filters :
        Key/value circuit-parameter filters applied at directory-scan time
        (same semantics as ``PlotConfig.filters``).
    group_by :
        Column names whose unique combinations produce separate bar groups on
        the same axes (e.g. ``["p"]`` for side-by-side bars per noise level).
    batch_indices :
        Restrict loading to specific batch indices (1-based integers matching
        ``batch=N`` in parquet filenames). ``None`` loads all available batches.
    normalize :
        When ``True``, show fractions (each group sums to 1).
        When ``False`` (default), show raw shot counts.
    include_negative_one :
        When ``True``, include case −1 (normal success) alongside cases 0–3.
        When ``False`` (default), show only the failure-analysis cases 0–3.
    """

    decoder_names: Optional[List[str]] = None
    filters: dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    batch_indices: Optional[List[int]] = None
    normalize: bool = False
    include_negative_one: bool = False


# ---------------------------------------------------------------------------
# Internal data loader
# ---------------------------------------------------------------------------

def _load_case_lazy(
    result_dir_root: Path,
    config: ForcedGapMLCaseConfig,
) -> pl.LazyFrame:
    """Return a concatenated LazyFrame of ``forced_gap_ml_case`` data.

    Searches decoder directories whose name encodes ``metric=forced_gap_ml``,
    then scans ``metric=forced_gap_ml_case_batch=*.parquet`` files inside them.
    The corresponding ``logicalerror_batch=*.parquet`` files are inner-joined on
    ``shot_id`` so that ``is_logical_error`` is always present in the result.

    Raises
    ------
    FileNotFoundError
        If no matching ``forced_gap_ml_case`` parquet files are found under
        *result_dir_root*.  This typically means the simulation was not run
        with ``get_all_failure_rate=True``.
    """
    frames: list[pl.LazyFrame] = []

    for circuit_entry in sorted(result_dir_root.iterdir()):
        if not circuit_entry.is_dir():
            continue
        if not _is_circuit_params_dir(circuit_entry.name):
            continue
        circuit_params = _parse_kv(circuit_entry.name)
        if not _matches_all_filters(circuit_params, config.filters):
            continue

        decoding_dir = circuit_entry / "decoding_result"
        if not decoding_dir.is_dir():
            continue

        for dm_entry in sorted(decoding_dir.iterdir()):
            if not dm_entry.is_dir():
                continue
            dm_params = _parse_kv(dm_entry.name)

            # Directory must encode metric=forced_gap_ml (primary metric name)
            if _strip_quotes(dm_params.get("metric", "")) != "forced_gap_ml":
                continue

            # Optional decoder name filter
            decoder = _strip_quotes(dm_params.get("decoder", ""))
            if config.decoder_names is not None and decoder not in config.decoder_names:
                continue

            # Collect per-batch forced_gap_ml_case parquet files
            for case_file in sorted(
                dm_entry.glob("metric=forced_gap_ml_case_batch=*.parquet")
            ):
                batch_idx = _extract_batch_idx(case_file.name)
                if batch_idx is None:
                    continue
                if (
                    config.batch_indices is not None
                    and batch_idx not in config.batch_indices
                ):
                    continue

                le_file = dm_entry / f"logicalerror_batch={batch_idx}.parquet"
                if not le_file.exists():
                    continue

                case_lf = pl.scan_parquet(case_file)
                le_lf = pl.scan_parquet(le_file)
                combined = case_lf.join(le_lf, on="shot_id", how="inner")

                all_params = {**circuit_params, **dm_params}
                literal_exprs = [
                    _infer_literal(val).alias(key) for key, val in all_params.items()
                ]
                literal_exprs.append(pl.lit(batch_idx, dtype=pl.Int64).alias("batch"))
                frames.append(combined.with_columns(literal_exprs))

    if not frames:
        raise FileNotFoundError(
            "No forced_gap_ml_case data found for the given filters "
            f"under {result_dir_root}. "
            "Make sure the simulation was run with get_all_failure_rate=True "
            "in the metric_options."
        )

    return pl.concat(frames, how="diagonal")


def _load_gap_and_case_lazy(
    result_dir_root: Path,
    config: GapCaseHistogramConfig,
) -> pl.LazyFrame:
    """Load forced_gap_ml and its corresponding case label joined on shot_id."""
    frames: list[pl.LazyFrame] = []

    for circuit_entry in sorted(result_dir_root.iterdir()):
        if not circuit_entry.is_dir() or not _is_circuit_params_dir(circuit_entry.name):
            continue
        circuit_params = _parse_kv(circuit_entry.name)
        if not _matches_all_filters(circuit_params, config.filters):
            continue

        decoding_dir = circuit_entry / "decoding_result"
        if not decoding_dir.is_dir():
            continue

        for dm_entry in sorted(decoding_dir.iterdir()):
            if not dm_entry.is_dir():
                continue
            dm_params = _parse_kv(dm_entry.name)

            if _strip_quotes(dm_params.get("metric", "")) != "forced_gap_ml":
                continue

            decoder = _strip_quotes(dm_params.get("decoder", ""))
            if config.decoder_names is not None and decoder not in config.decoder_names:
                continue

            for case_file in sorted(dm_entry.glob("metric=forced_gap_ml_case_batch=*.parquet")):
                batch_idx = _extract_batch_idx(case_file.name)
                if batch_idx is None:
                    continue
                if config.batch_indices is not None and batch_idx not in config.batch_indices:
                    continue

                fg_file = dm_entry / f"metric=forced_gap_ml_batch={batch_idx}.parquet"
                if not fg_file.exists():
                    continue

                combined = (
                    pl.scan_parquet(case_file)
                    .join(pl.scan_parquet(fg_file), on="shot_id", how="inner")
                )

                all_params = {**circuit_params, **dm_params}
                literal_exprs = [
                    _infer_literal(val).alias(key) for key, val in all_params.items()
                ]
                literal_exprs.append(pl.lit(batch_idx, dtype=pl.Int64).alias("batch"))
                frames.append(combined.with_columns(literal_exprs))

    if not frames:
        raise FileNotFoundError(
            "No matched gap and case data found. Check filters and simulation outputs."
        )

    return pl.concat(frames, how="diagonal")



# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ForcedGapMLCaseAnalyzer:
    """Bar-chart analyzer for the forced_gap_ml_case label distribution.

    Reads ``metric=forced_gap_ml_case_batch=*.parquet`` files produced when
    the :class:`ForcedGapMLDecoder` is run with ``get_all_failure_rate=True``,
    and draws a grouped bar chart showing how many shots (or what fraction)
    fall into each case label.

    Case labels
    -----------
    ===  =================================================================
     −1  Normal success — Stage 1 correct, adopted.
      0  All K+1 solutions are logical errors.
      1  Stage 2 rescue — Stage 1 error; correct Stage 2 solution adopted.
      2  Missed rescue  — Stage 1 error; correct Stage 2 existed but was
         not adopted (not minimum weight).
      3  Stage 1 overridden — Stage 1 correct; lower-weight Stage 2 adopted.
    ===  =================================================================

    Example
    -------
    ::

        config = ForcedGapMLCaseConfig(
            decoder_names=["BP-LSD"],
            filters={"d": 6, "p": 0.02, ...},
            normalize=True,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ForcedGapMLCaseAnalyzer().plot_case_histogram(manager, config, ax)
        plt.show()
    """

    def plot_case_histogram(
        self,
        manager: SimulationDataManager,
        config: ForcedGapMLCaseConfig,
        ax: plt.Axes,
    ) -> None:
        """Draw the case-label bar chart onto *ax*.

        Parameters
        ----------
        manager :
            :class:`SimulationDataManager` pointing at the result root.
        config :
            Histogram configuration (filters, grouping, normalisation).
        ax :
            Matplotlib ``Axes`` to draw on.
        """
        lf = _load_case_lazy(manager.result_dir_root, config)
        case_values: list[int] = (
            [-1, 0, 1, 2, 3] if config.include_negative_one else [0, 1, 2, 3]
        )

        needed = ["forced_gap_ml_case"] + config.group_by
        schema = set(lf.collect_schema().names())
        existing = [c for c in needed if c in schema]
        df = lf.select(existing).collect()

        x_pos = np.arange(len(case_values), dtype=float)

        if not config.group_by:
            heights = self._compute_heights(df, case_values, config.normalize)
            bars = ax.bar(x_pos, heights, width=0.6, alpha=0.8)
            # Label each bar with its value
            for bar, h in zip(bars, heights):
                if h > 0:
                    fmt = f"{h:.3f}" if config.normalize else f"{int(h):,}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        fmt,
                        ha="center", va="bottom", fontsize=8,
                    )
        else:
            partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                config.group_by, as_dict=True
            )
            sorted_keys = sorted(partitions)
            n_groups = len(sorted_keys)
            width = 0.8 / n_groups
            offsets = (np.arange(n_groups, dtype=float) - (n_groups - 1) / 2.0) * width

            for offset, key_vals in zip(offsets, sorted_keys):
                group_label = ", ".join(
                    f"{k}={v}" for k, v in zip(config.group_by, key_vals)
                )
                heights = self._compute_heights(
                    partitions[key_vals], case_values, config.normalize
                )
                ax.bar(x_pos + offset, heights, width=width, label=group_label, alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([CASE_TICK_LABELS[v] for v in case_values], fontsize=9)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_heights(
        df: pl.DataFrame,
        case_values: list[int],
        normalize: bool,
    ) -> np.ndarray:
        col = df["forced_gap_ml_case"].cast(pl.Int16)
        total = len(col)
        counts = np.array([int((col == v).sum()) for v in case_values], dtype=float)
        if normalize and total > 0:
            counts /= total
        return counts

    # ==================================================================
    # Scatter-plot API
    # ==================================================================

    def plot_case_scatter(
        self,
        manager: SimulationDataManager,
        config: "CaseScatterConfig",
        ax: plt.Axes,
        *,
        scatter_kw: dict[str, Any] | None = None,
    ) -> plt.Axes:
        """Scatter plot of forced_gap_ml vs another metric, filtered by case label.

        Each point in the scatter corresponds to a single shot whose
        ``forced_gap_ml_case`` label is one of ``config.case_values``.
        The x-axis shows the ``forced_gap_ml`` value for that shot; the
        y-axis shows the value of ``config.other_metric`` (loaded from a
        potentially different decoder directory) for the *same* shot.

        When ``config.color_by_logical_error=True``, shots for which the
        *other-metric* decoder made a logical error are drawn with a
        different marker shape **and** colour from non-error shots.

        Parameters
        ----------
        manager :
            :class:`SimulationDataManager` pointing at the result root.
        config :
            Scatter-plot configuration.
        ax :
            Matplotlib ``Axes`` to draw on.
        scatter_kw :
            Extra keyword arguments forwarded to every
            :meth:`matplotlib.axes.Axes.scatter` call (e.g.
            ``{"alpha": 0.5, "s": 10}``).  Per-call overrides such as
            *color* or *marker* are applied internally.

        Returns
        -------
        plt.Axes
            The axes with the completed scatter plot.
        """
        scatter_kw = dict(scatter_kw or {})

        # ---- 1. Load forced_gap_ml + forced_gap_ml_case ----------------
        fg_lf = _load_forced_gap_with_case_lazy(manager.result_dir_root, config)

        # ---- 2. Load the other metric via standard query ---------------
        other_plot_cfg = PlotConfig(
            metric_name=config.other_metric,
            decoder_names=config.other_decoder_names,
            filters=config.filters,
            batch_indices=config.batch_indices,
        )
        other_lf = manager.query(other_plot_cfg)

        # ---- 3. Determine join keys ------------------------------------
        schema_fg = set(fg_lf.collect_schema().names())
        schema_other = set(other_lf.collect_schema().names())

        join_keys = ["shot_id"] + [
            k for k in _CIRCUIT_PARAM_KEYS if k in schema_fg and k in schema_other
        ]

        fg_cols = list(dict.fromkeys(
            [c for c in join_keys if c in schema_fg]
            + ["forced_gap_ml", "forced_gap_ml_case", "is_logical_error"]
            + [c for c in config.group_by if c in schema_fg]
        ))
        other_cols = list(dict.fromkeys(
            [c for c in join_keys if c in schema_other]
            + [config.other_metric, "is_logical_error"]
        ))

        fg_df = fg_lf.select([c for c in fg_cols if c in schema_fg]).collect()
        other_df = other_lf.select([c for c in other_cols if c in schema_other]).collect()

        # Rename is_logical_error to avoid collision after join
        if "is_logical_error" in fg_df.columns:
            fg_df = fg_df.rename({"is_logical_error": "is_logical_error_fg"})
        if "is_logical_error" in other_df.columns:
            other_df = other_df.rename({"is_logical_error": "is_logical_error_other"})

        actual_keys = [k for k in join_keys if k in fg_df.columns and k in other_df.columns]
        df = fg_df.join(other_df, on=actual_keys, how="inner")

        # ---- 4. Filter to requested case labels ------------------------
        df = df.filter(pl.col("forced_gap_ml_case").is_in(config.case_values))

        if df.is_empty():
            ax.text(
                0.5, 0.5,
                f"No data found for case(s) {config.case_values}",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
            )
            return ax

        # ---- 5. Draw scatter ------------------------------------------
        prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
        palette = prop_cycle.by_key().get("color", []) if prop_cycle else []
        if not palette:
            palette = [f"C{i}" for i in range(10)]

        if not config.group_by:
            non_err_color = (
                _NON_ERROR_COLORS[0] if config.color_by_logical_error
                else scatter_kw.pop("color", palette[0])
            )
            err_color = _ERROR_COLORS[0]
            self._draw_scatter(
                ax, df, config.other_metric, config.color_by_logical_error,
                label=None,
                non_err_color=non_err_color,
                err_color=err_color,
                base_kw=scatter_kw,
            )
        else:
            group_cols = [c for c in config.group_by if c in df.columns]
            partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                group_cols, as_dict=True
            )
            for idx, key_vals in enumerate(sorted(partitions)):
                group_label = ", ".join(
                    f"{k}={v}" for k, v in zip(group_cols, key_vals)
                )
                non_err_color = (
                    _NON_ERROR_COLORS[idx % len(_NON_ERROR_COLORS)]
                    if config.color_by_logical_error
                    else palette[idx % len(palette)]
                )
                err_color = _ERROR_COLORS[idx % len(_ERROR_COLORS)]
                self._draw_scatter(
                    ax, partitions[key_vals], config.other_metric,
                    config.color_by_logical_error,
                    label=group_label,
                    non_err_color=non_err_color,
                    err_color=err_color,
                    base_kw=scatter_kw,
                )

        return ax

    def plot_gap_case_histogram(
        self,
        manager: SimulationDataManager,
        config: GapCaseHistogramConfig,
        ax: plt.Axes,
    ) -> None:
        """Plot a stacked histogram of cases 0-3 over binned gap values.

        Optionally overlays case -1 as a scatter plot.
        """
        lf = _load_gap_and_case_lazy(manager.result_dir_root, config)
        
        # 必要なカラムの抽出と null の除外
        df = lf.select(["forced_gap_ml", "forced_gap_ml_case"]).drop_nulls().collect()
        if df.is_empty():
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center")
            return

        # gap 値の丸め (ビンの決定)
        df = df.with_columns(
            pl.col("forced_gap_ml").round(config.gap_round_digits).alias("gap_bin")
        )

        # 全ケースの集計
        grouped = df.group_by(["gap_bin", "forced_gap_ml_case"]).agg(pl.len().alias("count"))
        gap_bins = np.sort(grouped["gap_bin"].unique().to_numpy())
        bin_to_idx = {b: i for i, b in enumerate(gap_bins)}

        # ケースごとのカウント配列を初期化
        cases_to_stack = [0, 1, 2, 3]
        case_counts = {c: np.zeros(len(gap_bins)) for c in cases_to_stack}
        neg_one_counts = np.zeros(len(gap_bins))

        for row in grouped.iter_rows(named=True):
            b = row["gap_bin"]
            c = row["forced_gap_ml_case"]
            idx = bin_to_idx[b]
            if c in case_counts:
                case_counts[c][idx] = row["count"]
            elif c == -1:
                neg_one_counts[idx] = row["count"]

        # 積み上げ棒グラフの描画 (case 0~3)
        bottom = np.zeros(len(gap_bins))
        total_stack_counts = sum(case_counts.values())

        # プロット用の色設定
        case_colors = {
            0: "tab:red",    # All fail
            1: "tab:blue",   # Rescued
            2: "tab:orange", # Missed
            3: "tab:purple", # Overridden
        }

        # 棒グラフの幅の設定
        width = 0.8 if config.gap_round_digits == 0 else (10 ** -config.gap_round_digits) * 0.8

        for c in cases_to_stack:
            counts = case_counts[c]
            if config.normalize_bars:
                # 0割り回避
                safe_totals = np.where(total_stack_counts == 0, 1, total_stack_counts)
                heights = counts / safe_totals
            else:
                heights = counts
                
            label_str = CASE_DESCRIPTIONS[c].split(":")[1].strip()
            ax.bar(
                gap_bins, heights, width=width, bottom=bottom,
                color=case_colors[c], label=f"Case {c}: {label_str}", alpha=0.85
            )
            bottom += heights

        ax.set_xlabel(f"forced_gap_ml (rounded to {config.gap_round_digits} decimals)")
        ylabel = "Fraction (Cases 0-3)" if config.normalize_bars else "Count (Cases 0-3)"
        ax.set_ylabel(ylabel)

        # ケース -1 の散布図プロット
        if config.plot_negative_one and np.any(neg_one_counts > 0):
            # normalize_bars が有効な場合は、スケールが異なるため右側のY軸を使用
            ax_neg = ax.twinx() if config.normalize_bars else ax
            
            valid_idx = neg_one_counts > 0
            x_neg = gap_bins[valid_idx]
            y_neg = neg_one_counts[valid_idx]
            
            scatter_plot = ax_neg.scatter(
                x_neg, y_neg, marker="o", color="tab:green", edgecolor="black",
                label="Case -1: Success", zorder=3
            )
            
            if config.normalize_bars:
                ax_neg.set_ylabel("Count (Case -1)", color="tab:green")
                ax_neg.tick_params(axis='y', labelcolor="tab:green")
                # twinx の凡例を統合
                lines, labels = ax.get_legend_handles_labels()
                lines.append(scatter_plot)
                labels.append("Case -1: Success")
                ax.legend(lines, labels, loc="upper right", fontsize=8)
            else:
                ax.legend(loc="upper right", fontsize=8)
        else:
            ax.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------

    @staticmethod
    def _draw_scatter(
        ax: plt.Axes,
        df: pl.DataFrame,
        other_metric: str,
        color_by_logical_error: bool,
        *,
        label: str | None,
        non_err_color: str,
        err_color: str,
        base_kw: dict[str, Any],
    ) -> None:
        needed = ["forced_gap_ml", other_metric]
        if color_by_logical_error and "is_logical_error_other" in df.columns:
            needed.append("is_logical_error_other")

        sub = df.select([c for c in needed if c in df.columns]).drop_nulls(
            ["forced_gap_ml", other_metric]
        )
        if sub.is_empty():
            return

        x = sub["forced_gap_ml"].to_numpy().astype(float)
        y = sub[other_metric].to_numpy().astype(float)

        has_err_col = (
            color_by_logical_error and "is_logical_error_other" in sub.columns
        )

        if has_err_col:
            is_err = sub["is_logical_error_other"].to_numpy().astype(bool)

            # Non-error shots: circle, non_err_color
            kw_ok = {k: v for k, v in base_kw.items() if k not in ("color", "c", "marker")}
            kw_ok["color"] = non_err_color
            kw_ok["marker"] = "o"
            label_ok = f"{label} (no error)" if label else "no logical error"
            if x[~is_err].size > 0:
                ax.scatter(x[~is_err], y[~is_err], label=label_ok, **kw_ok)

            # Logical-error shots: X marker, err_color
            kw_err = {k: v for k, v in base_kw.items() if k not in ("color", "c", "marker")}
            kw_err["color"] = err_color
            kw_err["marker"] = "x"
            kw_err.setdefault("linewidths", 1.5)
            label_err = f"{label} (logical error)" if label else "logical error"
            if x[is_err].size > 0:
                ax.scatter(x[is_err], y[is_err], label=label_err, **kw_err)
        else:
            kw = {k: v for k, v in base_kw.items() if k not in ("color", "c", "marker")}
            kw["color"] = non_err_color
            kw.setdefault("marker", "o")
            ax.scatter(x, y, label=label, **kw)


# ---------------------------------------------------------------------------
# CaseScatterConfig (defined after ForcedGapMLCaseAnalyzer to allow forward ref)
# ---------------------------------------------------------------------------

@dataclass
class CaseScatterConfig:
    """Configuration for case-filtered forced_gap_ml vs other-metric scatter plots.

    Each plotted shot has a ``forced_gap_ml_case`` label in ``case_values``.
    The x-axis shows the shot's ``forced_gap_ml`` value; the y-axis shows
    the same shot's ``other_metric`` value (possibly from a different decoder).

    Parameters
    ----------
    case_values :
        Case label(s) to include (any subset of ``[-1, 0, 1, 2, 3]``).
    other_metric :
        Name of the second metric for the y-axis (e.g.
        ``"linearize_logicalgap"``).
    other_decoder_names :
        Decoder names used to load *other_metric*. ``None`` includes all.
    forced_gap_decoder_names :
        Decoder names used to find the ``forced_gap_ml`` / case files.
        ``None`` includes all.
    filters :
        Shared circuit-parameter filters (same semantics as
        ``PlotConfig.filters``).
    group_by :
        Column names for separate scatter series on the same axes (e.g.
        ``["d"]``).
    batch_indices :
        Restrict loading to specific batch indices. ``None`` loads all.
    color_by_logical_error :
        When ``True``, shots for which the *other-metric* decoder made a
        logical error are drawn with a **different colour and marker** from
        non-error shots (circle vs. cross; paired colour palettes).
    """

    case_values: list[int]
    other_metric: str
    other_decoder_names: list[str] | None = None
    forced_gap_decoder_names: list[str] | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    group_by: list[str] = field(default_factory=list)
    batch_indices: list[int] | None = None
    color_by_logical_error: bool = False


# ---------------------------------------------------------------------------
# Data loader for scatter plots (forced_gap_ml + case, joined on shot_id)
# ---------------------------------------------------------------------------

def _load_forced_gap_with_case_lazy(
    result_dir_root: Path,
    config: CaseScatterConfig,
) -> pl.LazyFrame:
    """Return a LazyFrame with forced_gap_ml, forced_gap_ml_case, and is_logical_error.

    Searches decoder directories whose name encodes ``metric=forced_gap_ml``
    and loads three files per batch — the primary metric file, the case file,
    and the logical-error file — joining all three on ``shot_id``.

    Raises
    ------
    FileNotFoundError
        If no matching files are found; typically means the simulation was
        not run with ``get_all_failure_rate=True``.
    """
    frames: list[pl.LazyFrame] = []

    for circuit_entry in sorted(result_dir_root.iterdir()):
        if not circuit_entry.is_dir():
            continue
        if not _is_circuit_params_dir(circuit_entry.name):
            continue
        circuit_params = _parse_kv(circuit_entry.name)
        if not _matches_all_filters(circuit_params, config.filters):
            continue

        decoding_dir = circuit_entry / "decoding_result"
        if not decoding_dir.is_dir():
            continue

        for dm_entry in sorted(decoding_dir.iterdir()):
            if not dm_entry.is_dir():
                continue
            dm_params = _parse_kv(dm_entry.name)

            if _strip_quotes(dm_params.get("metric", "")) != "forced_gap_ml":
                continue

            decoder = _strip_quotes(dm_params.get("decoder", ""))
            if (
                config.forced_gap_decoder_names is not None
                and decoder not in config.forced_gap_decoder_names
            ):
                continue

            for case_file in sorted(
                dm_entry.glob("metric=forced_gap_ml_case_batch=*.parquet")
            ):
                batch_idx = _extract_batch_idx(case_file.name)
                if batch_idx is None:
                    continue
                if (
                    config.batch_indices is not None
                    and batch_idx not in config.batch_indices
                ):
                    continue

                fg_file = dm_entry / f"metric=forced_gap_ml_batch={batch_idx}.parquet"
                le_file = dm_entry / f"logicalerror_batch={batch_idx}.parquet"
                if not fg_file.exists() or not le_file.exists():
                    continue

                # Join the three files on shot_id
                combined = (
                    pl.scan_parquet(case_file)
                    .join(pl.scan_parquet(fg_file), on="shot_id", how="inner")
                    .join(pl.scan_parquet(le_file), on="shot_id", how="inner")
                )

                all_params = {**circuit_params, **dm_params}
                literal_exprs = [
                    _infer_literal(val).alias(key) for key, val in all_params.items()
                ]
                literal_exprs.append(pl.lit(batch_idx, dtype=pl.Int64).alias("batch"))
                frames.append(combined.with_columns(literal_exprs))

    if not frames:
        raise FileNotFoundError(
            "No forced_gap_ml + forced_gap_ml_case data found for the given "
            f"filters under {result_dir_root}. "
            "Make sure the simulation was run with get_all_failure_rate=True."
        )

    return pl.concat(frames, how="diagonal")
