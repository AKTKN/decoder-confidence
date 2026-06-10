"""Relative-improvement post-selection plot.

For each comparison metric, computes

    η(r) = ρ_base(r) / ρ_metric(r)

where ρ = (post-selected LER) / (original LER) is the LER reduction rate
and *r* is the abort rate.  The y-axis shows log₁₀(η): a positive value
means the comparison metric achieves a *lower* reduction rate than the base
(i.e. better post-selection) at that abort rate.

Curve style:
- Continuous (numeric) metrics — smooth line, no markers.
- Boolean AR metrics (swept over *b*) — markers, one point per *b* value.

Confidence intervals are computed conservatively:

    CI_lo(η) = ρ_base_CI_lo / ρ_metric_CI_hi
    CI_hi(η) = ρ_base_CI_hi / ρ_metric_CI_lo

and converted to log₁₀ scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis.src.config import PlotConfig
from analysis.src.confidence import shade_ci
from analysis.src.data_manager import SimulationDataManager
from analysis.src.postselect import (
    _format_label,
    _infer_direction,
    postselect_curve_ar,
    postselect_curve_continuous,
)


@dataclass
class RelImprovSpec:
    """Spec for one metric in a relative improvement plot (base or comparison).

    Attributes
    ----------
    metric_name : str
        Name of the metric column.
    decoder_names : list[str] or None
        Optional decoder name filter.
    filters : dict
        Additional column filters passed to SimulationDataManager.
    group_by : list[str]
        Partition keys; one curve is drawn per unique combination.
    batch_indices : list[int] or None
        Optional batch index filter.
    direction : str or None
        ``"high"`` / ``"low"`` / ``"boolean"``.  Inferred from the metric name
        if omitted.
    label_prefix : str or None
        Text prepended to auto-generated legend labels.
    grid_scale : str
        ``"linear"`` (default) or ``"log"`` — quantile grid spacing for
        continuous metrics.  Ignored for boolean (AR) metrics.
    """

    metric_name: str
    decoder_names: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    batch_indices: Optional[List[int]] = None
    direction: Optional[str] = None
    label_prefix: Optional[str] = None
    grid_scale: str = "linear"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_df(
    manager: SimulationDataManager,
    spec: RelImprovSpec,
    direction: str,
) -> pl.DataFrame:
    lf = manager.query(PlotConfig(
        metric_name=spec.metric_name,
        decoder_names=spec.decoder_names,
        filters=spec.filters,
        group_by=spec.group_by,
        batch_indices=spec.batch_indices,
    ))
    needed = list(dict.fromkeys(
        [spec.metric_name, "is_logical_error"] + list(spec.group_by)
    ))
    if direction == "boolean":
        needed.append("b")
    schema_names = set(lf.collect_schema().names())
    existing = [c for c in needed if c in schema_names]
    return lf.select(existing).collect()


def _original_ler(df: pl.DataFrame) -> float:
    n = len(df)
    if n == 0:
        return np.nan
    return float(df["is_logical_error"].cast(pl.Float64).sum()) / n


def _rho_curve(
    df: pl.DataFrame,
    spec: RelImprovSpec,
    direction: str,
    num_points: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (abort_rates, rho, rho_ci_low, rho_ci_high, original_ler).

    rho = post_ler / original_ler.  Arrays are sorted by abort_rates.
    All output arrays are NaN-safe (NaN where statistics are unavailable).
    """
    original_ler = _original_ler(df)
    if np.isnan(original_ler) or original_ler == 0.0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, original_ler

    if direction == "boolean":
        curve = postselect_curve_ar(df, spec.metric_name, alpha=alpha)
    else:
        sub = df.drop_nulls([spec.metric_name, "is_logical_error"])
        values = sub[spec.metric_name].to_numpy().astype(float)
        is_error = sub["is_logical_error"].to_numpy().astype(bool)
        curve = postselect_curve_continuous(
            values, is_error, direction,
            num_points=num_points, alpha=alpha,
            grid_scale=spec.grid_scale,
        )

    rho = curve.post_lers / original_ler
    rho_lo = curve.ci_low / original_ler
    rho_hi = curve.ci_high / original_ler
    return curve.abort_rates, rho, rho_lo, rho_hi, original_ler


# ---------------------------------------------------------------------------
# Main plotter
# ---------------------------------------------------------------------------

class RelativeImprovementPlotter:
    """Plot log₁₀(η) = log₁₀(ρ_base / ρ_metric) vs abort rate.

    Parameters
    ----------
    manager : SimulationDataManager
    """

    def __init__(self, manager: SimulationDataManager) -> None:
        self.manager = manager

    def plot(
        self,
        base_spec: RelImprovSpec,
        comparison_specs: List[RelImprovSpec],
        ax: plt.Axes,
        num_points: int = 200,
        alpha: float = 0.05,
        shade_alpha: float = 0.2,
        **plot_kw: Any,
    ) -> None:
        """Draw relative improvement curves.

        Parameters
        ----------
        base_spec :
            Baseline metric.  Its ρ is the numerator of η.
        comparison_specs :
            Metrics to compare against the base.  One curve per spec (and per
            group when ``group_by`` is set).
        ax :
            Matplotlib axes to draw on.
        num_points :
            Grid resolution for continuous metrics.  A high value (≥ 100)
            ensures the interpolated base curve is smooth.
        alpha :
            Wilson CI significance level (default 0.05 → 95 % CI).
        shade_alpha :
            Opacity for CI shading.
        """
        # ── compute base ρ curve (high-res for accurate interpolation) ────
        base_direction = base_spec.direction or _infer_direction(base_spec.metric_name)
        base_df = _collect_df(self.manager, base_spec, base_direction)
        ar_b, rho_b, rho_b_lo, rho_b_hi, _ = _rho_curve(
            base_df, base_spec, base_direction, num_points, alpha
        )
        if ar_b.size == 0:
            return

        valid_b = ~np.isnan(rho_b)
        ar_b_v = ar_b[valid_b]
        rho_b_v = rho_b[valid_b]
        rho_b_lo_v = rho_b_lo[valid_b]
        rho_b_hi_v = rho_b_hi[valid_b]
        if ar_b_v.size == 0:
            return

        # ── comparison curves ─────────────────────────────────────────────
        for spec in comparison_specs:
            direction = spec.direction or _infer_direction(spec.metric_name)
            df = _collect_df(self.manager, spec, direction)
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
                    self._plot_one(
                        part_df, spec, direction,
                        ar_b_v, rho_b_v, rho_b_lo_v, rho_b_hi_v,
                        ax, num_points, alpha, shade_alpha, plot_kw, label,
                    )
            else:
                label = f"{prefix}{spec.metric_name}"
                self._plot_one(
                    df, spec, direction,
                    ar_b_v, rho_b_v, rho_b_lo_v, rho_b_hi_v,
                    ax, num_points, alpha, shade_alpha, plot_kw, label,
                )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    # ------------------------------------------------------------------

    def _plot_one(
        self,
        df: pl.DataFrame,
        spec: RelImprovSpec,
        direction: str,
        ar_b_v: np.ndarray,
        rho_b_v: np.ndarray,
        rho_b_lo_v: np.ndarray,
        rho_b_hi_v: np.ndarray,
        ax: plt.Axes,
        num_points: int,
        alpha: float,
        shade_alpha: float,
        plot_kw: dict,
        label: str,
    ) -> None:
        ar_m, rho_m, rho_m_lo, rho_m_hi, _ = _rho_curve(
            df, spec, direction, num_points, alpha
        )
        if ar_m.size == 0:
            return

        valid_m = ~np.isnan(rho_m)
        ar_m_v = ar_m[valid_m]
        rho_m_v = rho_m[valid_m]
        rho_m_lo_v = rho_m_lo[valid_m]
        rho_m_hi_v = rho_m_hi[valid_m]
        if ar_m_v.size == 0:
            return

        # Interpolate ρ_base at the comparison abort rates.
        # np.interp clamps at the boundary for out-of-range values.
        rho_b_at_m = np.interp(ar_m_v, ar_b_v, rho_b_v)
        rho_b_lo_at_m = np.interp(ar_m_v, ar_b_v, rho_b_lo_v)
        rho_b_hi_at_m = np.interp(ar_m_v, ar_b_v, rho_b_hi_v)

        with np.errstate(divide="ignore", invalid="ignore"):
            # Point estimate
            eta = np.where(rho_m_v > 0, rho_b_at_m / rho_m_v, np.nan)
            log_eta = np.where(eta > 0, np.log10(eta), np.nan)

            # Conservative CI bounds on η:
            #   η_lo = ρ_base_CI_lo / ρ_metric_CI_hi
            #   η_hi = ρ_base_CI_hi / ρ_metric_CI_lo
            eta_lo = np.where(rho_m_hi_v > 0, rho_b_lo_at_m / rho_m_hi_v, np.nan)
            eta_hi = np.where(rho_m_lo_v > 0, rho_b_hi_at_m / rho_m_lo_v, np.nan)
            log_eta_lo = np.where(eta_lo > 0, np.log10(eta_lo), np.nan)
            log_eta_hi = np.where(eta_hi > 0, np.log10(eta_hi), np.nan)

        valid = ~np.isnan(log_eta)
        if not valid.any():
            return

        use_markers = direction == "boolean"
        if use_markers:
            line, = ax.plot(
                ar_m_v[valid], log_eta[valid],
                marker="o", linestyle="-", label=label, **plot_kw,
            )
        else:
            line, = ax.plot(
                ar_m_v[valid], log_eta[valid],
                label=label, **plot_kw,
            )
        shade_ci(
            ax, ar_m_v, log_eta_lo, log_eta_hi,
            color=line.get_color(), alpha=shade_alpha,
        )
