"""Post-selection performance plots.

For each confidence metric, sweep an abort threshold to trace out the trade-off
between abort rate (fraction of shots discarded) and post-selected logical
error rate (LER measured on the accepted subset). Wilson confidence intervals
are shaded around each curve.

Direction depends on the metric:

* ``logical_gap``, ``linearize_logicalgap``: higher = more confident, so
  low-value shots are aborted preferentially (``direction="high"``).
* ``cluster_llr``: lower = more confident, so high-value shots are aborted
  preferentially (``direction="low"``).
* ``ar-pec`` / ``ar-lec``: accept/reject is already decided per-shot; the
  curve is swept across the reweighting parameter ``b`` (one point per ``b``).

Multiple metrics can be plotted on the same axes by passing several
:class:`PostSelectSpec` instances to :meth:`PostSelectionPlotter.plot`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis.src.config import PlotConfig
from analysis.src.confidence import shade_ci, wilson_ci
from analysis.src.data_manager import SimulationDataManager


HIGH_CONFIDENCE_METRICS: frozenset[str] = frozenset(
    {"logical_gap", "linearize_logicalgap"}
)
LOW_CONFIDENCE_METRICS: frozenset[str] = frozenset({"cluster_llr"})
BOOLEAN_METRICS: frozenset[str] = frozenset({"ar-pec", "ar-lec"})


def _infer_direction(metric_name: str) -> str:
    if metric_name in HIGH_CONFIDENCE_METRICS:
        return "high"
    if metric_name in LOW_CONFIDENCE_METRICS:
        return "low"
    if metric_name in BOOLEAN_METRICS:
        return "boolean"
    raise ValueError(
        f"Unknown metric '{metric_name}' — register it in "
        f"HIGH_CONFIDENCE_METRICS / LOW_CONFIDENCE_METRICS / BOOLEAN_METRICS "
        f"or pass direction= explicitly on the spec."
    )


def _format_label(parts: list[tuple[str, Any]]) -> str:
    return ", ".join(f"{k}={v}" for k, v in parts)


@dataclass
class PostSelectSpec:
    """One metric's contribution to a post-selection plot.

    Attributes
    ----------
    threshold_mode:
        ``"continuous"`` (default) sweeps a quantile grid of *num_points*
        abort-rate values.  ``"unique"`` uses every distinct metric value as
        a threshold (one curve point per unique value); markers are drawn in
        this mode.
    grid_scale:
        For ``threshold_mode="continuous"`` only.  ``"linear"`` (default) for
        uniformly spaced quantiles; ``"log"`` to concentrate points near
        abort-rate = 0.
    """

    metric_name: str
    decoder_names: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    batch_indices: Optional[List[int]] = None
    direction: Optional[str] = None
    label_prefix: Optional[str] = None
    threshold_mode: str = "continuous"
    grid_scale: str = "linear"


@dataclass
class PostSelectCurve:
    """Per-point post-selection statistics."""

    abort_rates: np.ndarray
    post_lers: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    accepted: np.ndarray


def postselect_curve_continuous(
    values: np.ndarray,
    is_error: np.ndarray,
    direction: str,
    num_points: int = 50,
    alpha: float = 0.05,
    grid_scale: str = "linear",
) -> PostSelectCurve:
    """Sweep an abort-rate grid by quantiles of *values* and compute post-LER.

    Parameters
    ----------
    values:
        Per-shot metric values.
    is_error:
        Per-shot boolean (or 0/1) array, aligned with *values*.
    direction:
        ``"high"`` to keep high values (abort low), ``"low"`` to keep low values
        (abort high).
    num_points:
        Number of points on the curve.
    alpha:
        Wilson CI significance level (default 0.05 → 95 % CI).
    grid_scale:
        ``"linear"`` (default) for uniformly spaced quantile grid, or ``"log"``
        to concentrate points near abort-rate = 0 (log-spaced quantiles).
    """
    if direction not in {"high", "low"}:
        raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")
    if grid_scale not in {"linear", "log"}:
        raise ValueError(f"grid_scale must be 'linear' or 'log', got {grid_scale!r}")

    values = np.asarray(values, dtype=float)
    is_error = np.asarray(is_error, dtype=bool)

    n_total = values.size
    if n_total == 0:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        return PostSelectCurve(empty_f, empty_f, empty_f, empty_f, empty_i)

    if grid_scale == "log":
        # Concentrate points near abort-rate = 0; include 0 explicitly.
        log_part = np.logspace(-3, np.log10(0.9999), num_points - 1)
        abort_grid = np.concatenate([[0.0], log_part])
    else:
        abort_grid = np.linspace(0.0, 1.0, num_points + 1)[:-1]

    abort_rates = np.full(num_points, np.nan)
    post_lers = np.full(num_points, np.nan)
    ci_low = np.full(num_points, np.nan)
    ci_high = np.full(num_points, np.nan)
    accepted = np.zeros(num_points, dtype=int)

    for i, r in enumerate(abort_grid):
        if direction == "high":
            threshold = np.quantile(values, r)
            mask = values >= threshold
        else:
            threshold = np.quantile(values, 1.0 - r)
            mask = values <= threshold

        n_acc = int(mask.sum())
        accepted[i] = n_acc
        abort_rates[i] = 1.0 - n_acc / n_total

        if n_acc == 0:
            continue

        k_err = int(is_error[mask].sum())
        post_lers[i] = k_err / n_acc
        lo, hi = wilson_ci(k_err, n_acc, alpha=alpha)
        ci_low[i] = float(lo)
        ci_high[i] = float(hi)

    order = np.argsort(abort_rates)
    return PostSelectCurve(
        abort_rates=abort_rates[order],
        post_lers=post_lers[order],
        ci_low=ci_low[order],
        ci_high=ci_high[order],
        accepted=accepted[order],
    )


def postselect_curve_unique_thresholds(
    values: np.ndarray,
    is_error: np.ndarray,
    direction: str,
    alpha: float = 0.05,
) -> PostSelectCurve:
    """Compute one post-selection point per unique metric value used as threshold.

    Parameters
    ----------
    values:
        Per-shot metric values.
    is_error:
        Per-shot boolean (or 0/1) array, aligned with *values*.
    direction:
        ``"high"`` to keep high values (abort low), ``"low"`` to keep low values
        (abort high).
    alpha:
        Wilson CI significance level (default 0.05 → 95 % CI).
    """
    if direction not in {"high", "low"}:
        raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")

    values = np.asarray(values, dtype=float)
    is_error = np.asarray(is_error, dtype=bool)
    n_total = values.size

    if n_total == 0:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        return PostSelectCurve(empty_f, empty_f, empty_f, empty_f, empty_i)

    unique_vals = np.unique(values)

    rows = []
    for threshold in unique_vals:
        if direction == "high":
            mask = values >= threshold
        else:
            mask = values <= threshold

        n_acc = int(mask.sum())
        if n_acc == 0:
            continue

        abort = 1.0 - n_acc / n_total
        k_err = int(is_error[mask].sum())
        post_ler = k_err / n_acc
        lo, hi = wilson_ci(k_err, n_acc, alpha=alpha)
        rows.append((abort, post_ler, float(lo), float(hi), n_acc))

    if not rows:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        return PostSelectCurve(empty_f, empty_f, empty_f, empty_f, empty_i)

    rows.sort(key=lambda r: r[0])
    abort_arr, ler_arr, lo_arr, hi_arr, acc_arr = zip(*rows)
    return PostSelectCurve(
        abort_rates=np.array(abort_arr, dtype=float),
        post_lers=np.array(ler_arr, dtype=float),
        ci_low=np.array(lo_arr, dtype=float),
        ci_high=np.array(hi_arr, dtype=float),
        accepted=np.array(acc_arr, dtype=int),
    )


def postselect_curve_ar(
    df: pl.DataFrame,
    metric_name: str,
    alpha: float = 0.05,
) -> PostSelectCurve:
    """Compute one post-selection point per distinct value of ``b``.

    Parameters
    ----------
    df:
        Collected DataFrame containing columns ``metric_name`` (bool accept
        flag), ``is_logical_error`` (bool), and ``b`` (float).
    metric_name:
        Name of the accept-flag column (e.g. ``"ar-pec"``).
    alpha:
        Wilson CI significance level.
    """
    required = {metric_name, "is_logical_error", "b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"AR post-selection requires columns {required}; missing {missing}"
        )

    rows = []
    parts: dict[tuple, pl.DataFrame] = df.partition_by(["b"], as_dict=True)
    for key_vals, part in parts.items():
        b_val = key_vals[0] if isinstance(key_vals, tuple) else key_vals
        accept = part[metric_name].to_numpy().astype(bool)
        is_err = part["is_logical_error"].to_numpy().astype(bool)

        n_total = accept.size
        n_acc = int(accept.sum())
        if n_total == 0:
            continue

        abort = 1.0 - n_acc / n_total
        if n_acc == 0:
            rows.append((b_val, abort, np.nan, np.nan, np.nan, 0))
            continue

        k_err = int(is_err[accept].sum())
        post_ler = k_err / n_acc
        lo, hi = wilson_ci(k_err, n_acc, alpha=alpha)
        rows.append((b_val, abort, post_ler, float(lo), float(hi), n_acc))

    if not rows:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        return PostSelectCurve(empty_f, empty_f, empty_f, empty_f, empty_i)

    rows.sort(key=lambda r: r[1])
    _, abort, ler, lo, hi, acc = zip(*rows)
    return PostSelectCurve(
        abort_rates=np.array(abort, dtype=float),
        post_lers=np.array(ler, dtype=float),
        ci_low=np.array(lo, dtype=float),
        ci_high=np.array(hi, dtype=float),
        accepted=np.array(acc, dtype=int),
    )


class PostSelectionPlotter:
    """Plot one or more metrics' post-selection curves on a shared Axes."""

    def __init__(self, manager: SimulationDataManager) -> None:
        self.manager = manager

    def plot(
        self,
        specs: List[PostSelectSpec],
        ax: plt.Axes,
        num_points: int = 50,
        alpha: float = 0.05,
        shade_alpha: float = 0.2,
        reduction_rate: bool = False,
        **plot_kw: Any,
    ) -> None:
        for spec in specs:
            self._plot_spec(
                spec, ax, num_points, alpha, shade_alpha, plot_kw, reduction_rate
            )

        ax.set_xlabel("Abort rate")
        if reduction_rate:
            ax.set_ylabel("LER reduction rate (post-LER / original-LER)")
        else:
            ax.set_ylabel("Post-selected logical error rate")
        ax.set_xlim(0.0, 1.0)
        ax.legend()

    def _plot_spec(
        self,
        spec: PostSelectSpec,
        ax: plt.Axes,
        num_points: int,
        alpha: float,
        shade_alpha: float,
        plot_kw: dict,
        reduction_rate: bool,
    ) -> None:
        direction = spec.direction or _infer_direction(spec.metric_name)

        plot_config = PlotConfig(
            metric_name=spec.metric_name,
            decoder_names=spec.decoder_names,
            filters=spec.filters,
            group_by=spec.group_by,
            batch_indices=spec.batch_indices,
        )
        lf = self.manager.query(plot_config)

        needed = [spec.metric_name, "is_logical_error"] + list(spec.group_by)
        if direction == "boolean":
            needed.append("b")
        schema_names = set(lf.collect_schema().names())
        existing = [c for c in dict.fromkeys(needed) if c in schema_names]
        df = lf.select(existing).collect()

        prefix = f"{spec.label_prefix} " if spec.label_prefix else ""

        if spec.group_by:
            partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                spec.group_by, as_dict=True
            )
            for key_vals in sorted(partitions):
                part_df = partitions[key_vals]
                key_tuple = key_vals if isinstance(key_vals, tuple) else (key_vals,)
                group_label = _format_label(list(zip(spec.group_by, key_tuple)))
                label = f"{prefix}{spec.metric_name} | {group_label}"
                self._compute_and_plot(
                    part_df, spec, direction, ax,
                    num_points, alpha, shade_alpha, plot_kw, label, reduction_rate,
                )
        else:
            label = f"{prefix}{spec.metric_name}"
            self._compute_and_plot(
                df, spec, direction, ax,
                num_points, alpha, shade_alpha, plot_kw, label, reduction_rate,
            )

    def _compute_and_plot(
        self,
        df: pl.DataFrame,
        spec: PostSelectSpec,
        direction: str,
        ax: plt.Axes,
        num_points: int,
        alpha: float,
        shade_alpha: float,
        plot_kw: dict,
        label: str,
        reduction_rate: bool,
    ) -> None:
        if direction == "boolean":
            curve = postselect_curve_ar(df, spec.metric_name, alpha=alpha)
            original_ler = df["is_logical_error"].mean()
        else:
            sub = df.drop_nulls([spec.metric_name, "is_logical_error"])
            values = sub[spec.metric_name].to_numpy().astype(float)
            is_error = sub["is_logical_error"].to_numpy().astype(bool)
            if spec.threshold_mode == "unique":
                curve = postselect_curve_unique_thresholds(
                    values, is_error, direction, alpha=alpha,
                )
            else:
                curve = postselect_curve_continuous(
                    values, is_error, direction,
                    num_points=num_points, alpha=alpha,
                    grid_scale=spec.grid_scale,
                )
            original_ler = float(is_error.mean()) if is_error.size > 0 else np.nan

        if curve.abort_rates.size == 0:
            return

        y = curve.post_lers
        ci_lo = curve.ci_low
        ci_hi = curve.ci_high

        if reduction_rate:
            if original_ler and original_ler > 0:
                y = y / original_ler
                ci_lo = ci_lo / original_ler
                ci_hi = ci_hi / original_ler
            else:
                return

        valid = ~np.isnan(y)
        use_markers = direction == "boolean" or spec.threshold_mode == "unique"
        if use_markers:
            line, = ax.plot(
                curve.abort_rates[valid], y[valid],
                marker="o", linestyle="-", label=label, **plot_kw,
            )
        else:
            line, = ax.plot(
                curve.abort_rates[valid], y[valid],
                label=label, **plot_kw,
            )
        shade_ci(
            ax, curve.abort_rates, ci_lo, ci_hi,
            color=line.get_color(), alpha=shade_alpha,
        )
