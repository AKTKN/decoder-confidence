"""Diagnostics for why anchored forced gap outperforms ordinary forced gap.

Background
----------
For each shot, the baseline decode produces a correction ``e^(0)`` with
weight ``w0`` and logical label ``lambda0``.  The forced-gap decoder
additionally produces ``K`` forced-run candidates ``e^(1), ..., e^(K)``
(one per logical observable), with weights ``w1, ..., wK``.

Two confidence metrics / final-answer strategies are built from these
candidates:

* **Ordinary forced gap** ``Delta_fg`` re-selects the final correction as
  the minimum-weight solution among ``{e^(0), e^(1), ..., e^(K)}`` and
  reports the gap to the next-best *distinct-logical-class* solution.
  This is exactly the ``forced_gap_ml`` metric/decoder output: its own
  ``is_logical_error`` column reflects the *reselected* answer, i.e.
  ``Y_star = 1[lambda_star != lambda_true]``.
* **Anchored forced gap** ``Delta_anc = min_i(w_i) - w0`` always keeps the
  baseline ``e^(0)`` as the final answer; the forced runs are only used to
  score confidence.  This quantity is numerically identical to the
  ``linearize_logicalgap`` metric, and can equivalently be derived from the
  ``forced_gap_ml`` detailed-stat columns as
  ``forced_stage2_weight - forced_stage1_weight`` (both are computed from
  the very same baseline/forced decodes).  This module takes the latter
  route so that only one decoder-result directory (``forced_gap_ml`` with
  ``get_detail_stat=True``) is required.

Given this single source table, per shot we have:

* ``y0``       -- ``Y_0 = 1[lambda0 != lambda_true]``      (baseline error)
* ``y_star``   -- ``Y_star = 1[lambda_star != lambda_true]`` (reselected error)
* ``delta_fg``  -- the ordinary forced gap ``Delta_fg``
* ``delta_anc`` -- the anchored forced gap ``Delta_anc``

:func:`compute_reselection_transition_by_forced_gap_bin` and
:func:`plot_reselection_transition` implement diagnostic plot (a): how the
ordinary forced-gap reselection changes the hard logical decision on shots
where a lighter forced candidate exists (``Delta_anc < 0``).

:func:`compute_postselection_decomposition_curve` and
:func:`plot_postselection_decomposition` implement diagnostic plot (b): the
decomposition of the post-selected LER advantage of anchored forced gap over
ordinary forced gap into a "decision effect" (impact of reselecting the
final correction) and a "ranking effect" (impact of which shots get
rejected).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis.src.confidence import bin_proportions, wilson_ci
from analysis.src.data_manager import (
    SimulationDataManager,
    _extract_batch_idx,
    _is_circuit_params_dir,
    _matches_all_filters,
    _parse_kv,
)
from analysis.src.ranking_degradation import _topk_accept_mask
from analysis.src.stage_gap_difference import _find_detail_dirs, _load_metric_detail_lazy

TransitionCategory = Literal["fixed", "broken", "both_wrong", "both_correct"]

#: Stacking order used by :func:`plot_reselection_transition` (bottom to top).
DEFAULT_CATEGORY_ORDER: tuple[TransitionCategory, ...] = (
    "both_correct",
    "fixed",
    "broken",
    "both_wrong",
)

_CATEGORY_LABELS: dict[str, str] = {
    "fixed": r"Fixed by reselecting ($Y_0{=}1,Y_*{=}0$)",
    "broken": r"Broken by reselecting ($Y_0{=}0,Y_*{=}1$)",
    "both_wrong": r"Both wrong ($Y_0{=}1,Y_*{=}1$)",
    "both_correct": r"Both correct ($Y_0{=}0,Y_*{=}0$)",
}

_CATEGORY_COLORS: dict[str, str] = {
    "both_correct": "tab:blue",
    "fixed": "tab:green",
    "broken": "tab:red",
    "both_wrong": "tab:orange",
}


@dataclass
class AnchoredReselectionConfig:
    """Configuration for loading ``forced_gap_ml`` detailed shot data.

    Parameters
    ----------
    decoder_names:
        Decoder directory names to include (e.g. ``["BP-LSD"]``). ``None``
        includes every decoder that has a ``metric=forced_gap_ml`` directory
        with ``get_detail_stat=True``.
    filters:
        Key/value conditions applied to circuit-parameter directory names,
        e.g. ``{"code": "bivariate_bicycle_code_Z", "d": 6, "p": 0.003}``.
    batch_indices:
        Optional one-based batch indices to restrict loading to. ``None``
        loads every available batch.
    """

    decoder_names: list[str] | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    batch_indices: list[int] | None = None


def load_anchored_reselection_lazy(
    result_dir_root: Path,
    config: AnchoredReselectionConfig,
) -> pl.LazyFrame:
    """Load per-shot ``forced_gap_ml`` detail data as a Polars LazyFrame.

    Only the ``forced_gap_ml`` decoder directory is scanned (with
    ``get_detail_stat=True``); no join with a separate ``linearize_logicalgap``
    or ``logical_gap`` decoder is required (see module docstring).

    Parameters
    ----------
    result_dir_root:
        Root directory containing circuit-parameter subdirectories.
    config:
        Directory / decoder / batch filters.

    Returns
    -------
    polars.LazyFrame
        Columns include ``shot_id``, ``batch``, ``forced_gap_ml``,
        ``forced_is_logical_error``, ``forced_stage1_weight``,
        ``forced_stage1_obs_flip``, ``forced_stage2_weight``,
        ``forced_stage2_obs_flip``, plus path-encoded circuit/decoder
        parameters.

    Raises
    ------
    FileNotFoundError
        If no matching ``forced_gap_ml`` detail data is found.
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

        for dm_dir, dm_params in _find_detail_dirs(
            circuit_dir, "forced_gap_ml", config.decoder_names
        ):
            all_params = {**circuit_params, **dm_params}
            for metric_file in sorted(dm_dir.glob("metric=forced_gap_ml_batch=*.parquet")):
                batch_idx = _extract_batch_idx(metric_file.name)
                if batch_idx is None:
                    continue
                if config.batch_indices is not None and batch_idx not in config.batch_indices:
                    continue
                frames.append(
                    _load_metric_detail_lazy(
                        dm_dir, "forced_gap_ml", batch_idx, all_params, prefix="forced"
                    )
                )

    if not frames:
        raise FileNotFoundError(
            "No forced_gap_ml detailed-stat data found "
            f"with filters={config.filters} under {root}. "
            "Data must come from a directory produced with "
            "metric=forced_gap_ml,get_detail_stat=True."
        )
    return pl.concat(frames, how="diagonal")


def _with_shot_indicator_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Attach ``y0``, ``y_star``, ``delta_fg``, ``delta_anc`` derived columns.

    See the module docstring for the correspondence between these names and
    the ``Y_0``, ``Y_*``, ``Delta_fg``, ``Delta_anc`` quantities.
    """
    return lf.with_columns(
        [
            pl.col("forced_stage1_obs_flip").alias("y0"),
            pl.col("forced_is_logical_error").alias("y_star"),
            pl.col("forced_gap_ml").alias("delta_fg"),
            (pl.col("forced_stage2_weight") - pl.col("forced_stage1_weight")).alias(
                "delta_anc"
            ),
        ]
    )


def collect_anchored_reselection_shot_table(
    manager: SimulationDataManager,
    config: AnchoredReselectionConfig,
) -> pl.DataFrame:
    """Return the per-shot table with ``y0``, ``y_star``, ``delta_fg``, ``delta_anc``.

    Example
    -------
    >>> config = AnchoredReselectionConfig(
    ...     decoder_names=["BP-LSD"], filters={"d": 6, "p": 0.003},
    ... )
    >>> df = collect_anchored_reselection_shot_table(manager, config)
    """
    lf = load_anchored_reselection_lazy(manager.result_dir_root, config)
    return _with_shot_indicator_columns(lf).collect()


def compute_negative_anchored_rate(df: pl.DataFrame) -> float:
    """Return ``r_minus = P(Delta_anc < 0)`` for the collected shot table."""
    if df.height == 0:
        return float("nan")
    return float((df["delta_anc"].to_numpy() < 0).mean())


def _category_masks(y0: np.ndarray, y_star: np.ndarray) -> dict[TransitionCategory, np.ndarray]:
    return {
        "fixed": y0 & ~y_star,
        "broken": ~y0 & y_star,
        "both_wrong": y0 & y_star,
        "both_correct": ~y0 & ~y_star,
    }


BinScale = Literal["gap_value", "abort_rate"]


def _gap_bin_edges(
    values: np.ndarray,
    bins: int | Sequence[float],
    x_range: tuple[float, float] | None = None,
) -> np.ndarray:
    if isinstance(bins, int):
        lo, hi = x_range if x_range is not None else (float(values.min()), float(values.max()))
        return np.linspace(lo, hi, bins + 1)
    return np.asarray(bins, dtype=float)


def _forced_gap_abort_rate(delta_fg_subset: np.ndarray, delta_fg_full: np.ndarray) -> np.ndarray:
    """Map forced-gap values to the abort rate at which they would first be rejected.

    Ordinary forced gap rejects the lowest-``Delta_fg`` shots first (this is
    the same "reject lowest" convention used by ``R_fg(r)`` in
    :func:`compute_postselection_decomposition_curve`), so the abort rate
    associated with a gap value ``g`` is defined as the fraction of the
    *full* shot population (not just the negative-anchored subset) with
    ``Delta_fg <= g``.  This lets plot (a) be read on the same rejection-rate
    axis as plot (b).
    """
    sorted_full = np.sort(delta_fg_full)
    n_total = delta_fg_full.size
    ranks = np.searchsorted(sorted_full, delta_fg_subset, side="right")
    return ranks / n_total


def compute_reselection_transition_by_forced_gap_bin(
    manager: SimulationDataManager,
    config: AnchoredReselectionConfig,
    *,
    bins: int | Sequence[float] = 10,
    alpha: float = 0.05,
    bin_scale: BinScale = "gap_value",
    x_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Compute transition-category fractions within ``A = {Delta_anc < 0}``.

    Restricts to the negative-anchored subset ``A`` and bins shots by the
    ordinary forced gap ``Delta_fg`` (or, when ``bin_scale="abort_rate"``, by
    the corresponding forced-gap abort rate -- see
    :func:`_forced_gap_abort_rate`).  Within each bin, returns the
    conditional fraction of four transition categories (they sum to 1 per
    bin): ``"fixed"`` (``Y_0=1,Y_*=0``), ``"broken"`` (``Y_0=0,Y_*=1``),
    ``"both_wrong"`` (``Y_0=1,Y_*=1``), and ``"both_correct"``
    (``Y_0=0,Y_*=0``).

    Parameters
    ----------
    manager:
        Data manager used to load the shot table.
    config:
        Loading configuration (filters / decoder / batch selection).
    bins:
        Either the number of equal-width bins spanning ``x_range`` (or, when
        ``x_range`` is ``None``, the observed range of the chosen
        ``bin_scale`` within ``A``), or explicit bin edges. When ``bins`` is
        an explicit edge sequence, ``x_range`` is ignored.
    alpha:
        Wilson CI significance level (default 0.05 -> 95% CI).
    bin_scale:
        ``"gap_value"`` (default) bins by the raw ``Delta_fg`` weight-gap
        value.  ``"abort_rate"`` bins by the fraction of the *full* shot
        population (i.e. ``r = P(Delta_fg <= g)``) that ordinary forced gap
        would reject at that gap value, so the x-axis is directly
        comparable to the rejection-rate axis of
        :func:`compute_postselection_decomposition_curve`.
    x_range:
        Optional ``(min, max)`` override for the binned range, in the units
        of ``bin_scale`` (only used when ``bins`` is an integer).  Shots
        outside this range are clamped into the first/last bin rather than
        dropped (same convention as
        :func:`analysis.src.confidence.bin_proportions`).  When ``None``,
        the range is taken from the data (``min``/``max`` of the chosen
        ``bin_scale`` within ``A``).

    Returns
    -------
    polars.DataFrame
        Long-form table with one row per ``(bin, category)`` pair and
        columns ``bin_low``, ``bin_high``, ``bin_center`` (in the units of
        ``bin_scale``), ``bin_scale``, ``category``, ``category_label``,
        ``n_bin`` (shots of ``A`` in that bin), ``fraction``, ``ci_low``,
        ``ci_high``.

    Raises
    ------
    ValueError
        If the negative-anchored subset ``A`` is empty, or ``bin_scale`` is
        not one of ``"gap_value"`` / ``"abort_rate"``.
    """
    if bin_scale not in ("gap_value", "abort_rate"):
        raise ValueError(
            f"bin_scale must be 'gap_value' or 'abort_rate', got {bin_scale!r}"
        )

    df = collect_anchored_reselection_shot_table(manager, config)
    a_mask = df["delta_anc"].to_numpy() < 0
    if not np.any(a_mask):
        raise ValueError(
            "Negative-anchored subset A = {Delta_anc < 0} is empty; "
            "cannot bin by forced gap."
        )

    delta_fg_full = df["delta_fg"].to_numpy().astype(float)
    sub = df.filter(pl.Series(a_mask))
    delta_fg = sub["delta_fg"].to_numpy().astype(float)
    y0 = sub["y0"].to_numpy().astype(bool)
    y_star = sub["y_star"].to_numpy().astype(bool)

    x = (
        _forced_gap_abort_rate(delta_fg, delta_fg_full)
        if bin_scale == "abort_rate"
        else delta_fg
    )

    edges = _gap_bin_edges(x, bins, x_range=x_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    categories = _category_masks(y0, y_star)

    rows: list[dict[str, Any]] = []
    for name, mask in categories.items():
        bstats = bin_proportions(x, mask.astype(float), bins=edges, alpha=alpha)
        for i in range(len(centers)):
            rows.append(
                {
                    "bin_low": float(edges[i]),
                    "bin_high": float(edges[i + 1]),
                    "bin_center": float(centers[i]),
                    "bin_scale": bin_scale,
                    "category": name,
                    "category_label": _CATEGORY_LABELS[name],
                    "n_bin": int(bstats.totals[i]),
                    "fraction": float(bstats.proportions[i]),
                    "ci_low": float(bstats.ci_low[i]),
                    "ci_high": float(bstats.ci_high[i]),
                }
            )
    return pl.DataFrame(rows)


def compute_negative_anchored_net_decision_effect(
    manager: SimulationDataManager,
    config: AnchoredReselectionConfig,
) -> dict[str, float]:
    """Return the net decision effect within ``A = {Delta_anc < 0}``.

    The net decision effect is
    ``P(Y_*=1|A) - P(Y_0=1|A) = P(Y_0=0,Y_*=1|A) - P(Y_0=1,Y_*=0|A)``.
    Positive means reselecting *worsens* the logical-error rate inside the
    negative-anchored subset; negative means it *improves* it.

    Returns
    -------
    dict
        Keys ``"n_A"``, ``"p_y0_given_A"``, ``"p_ystar_given_A"``,
        ``"net_decision_effect"``.
    """
    df = collect_anchored_reselection_shot_table(manager, config)
    a_mask = df["delta_anc"].to_numpy() < 0
    sub = df.filter(pl.Series(a_mask))
    n_a = sub.height
    if n_a == 0:
        nan = float("nan")
        return {
            "n_A": 0,
            "p_y0_given_A": nan,
            "p_ystar_given_A": nan,
            "net_decision_effect": nan,
        }

    p_y0 = float(sub["y0"].to_numpy().astype(bool).mean())
    p_ystar = float(sub["y_star"].to_numpy().astype(bool).mean())
    return {
        "n_A": n_a,
        "p_y0_given_A": p_y0,
        "p_ystar_given_A": p_ystar,
        "net_decision_effect": p_ystar - p_y0,
    }


def _default_abort_rate_grid(r_minus: float, num_points: int, max_multiple: float) -> np.ndarray:
    """Build a rejection-rate grid on ``[0, hi]`` that always includes ``r_minus``."""
    hi = min(0.95, r_minus * max_multiple) if r_minus > 0 else 0.5
    hi = max(hi, 1e-3)
    grid = np.linspace(0.0, hi, num_points)
    grid = np.append(grid, r_minus)
    grid = np.clip(grid, 0.0, 0.999999)
    return np.unique(grid)


def compute_postselection_decomposition_curve(
    manager: SimulationDataManager,
    config: AnchoredReselectionConfig,
    *,
    abort_rates: Sequence[float] | None = None,
    num_points: int = 30,
    max_multiple: float = 2.0,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Decompose ``L_fg(r) - L_anc(r)`` into decision and ranking effects.

    For a matched rejection rate ``r``, ``R_anc(r)`` / ``R_fg(r)`` reject the
    ``r``-fraction of shots with the lowest ``Delta_anc`` / ``Delta_fg``
    values respectively (both use the same top-k-by-score selection as
    :func:`analysis.src.ranking_degradation._topk_accept_mask`, so rejection
    rates are matched by construction).

    ``L_anc(r) = P(Y_0=1, not-R_anc(r)) / (1-r)`` and
    ``L_fg(r) = P(Y_*=1, not-R_fg(r)) / (1-r)`` are the post-selected LERs of
    the anchored and ordinary forced-gap strategies.  The difference
    decomposes as ``L_fg(r) - L_anc(r) = D_decision(r) + D_ranking(r)``:

    * ``D_decision(r)`` isolates the effect of reselecting the final
      correction (``e^(0) -> e^(*)``) on the shots accepted by ordinary
      forced gap.
    * ``D_ranking(r)`` isolates the effect of which shots get rejected,
      holding the baseline logical-error indicator ``Y_0`` fixed.

    Parameters
    ----------
    manager:
        Data manager used to load the shot table.
    config:
        Loading configuration.
    abort_rates:
        Explicit rejection-rate grid.  When ``None``, a grid on
        ``[0, min(0.95, max_multiple * r_minus)]`` is built automatically
        and ``r_minus = P(Delta_anc < 0)`` is included exactly (see
        :func:`_default_abort_rate_grid`).
    num_points:
        Number of points in the automatic grid (ignored if ``abort_rates``
        is given).
    max_multiple:
        How far past ``r_minus`` the automatic grid extends (ignored if
        ``abort_rates`` is given).
    alpha:
        Wilson CI significance level for ``L_anc`` / ``L_fg`` (default 0.05
        -> 95% CI).

    Returns
    -------
    polars.DataFrame
        Columns ``abort_rate``, ``r_minus`` (constant column, ``P(Delta_anc
        < 0)``), ``L_anc``, ``L_anc_ci_low``, ``L_anc_ci_high``, ``L_fg``,
        ``L_fg_ci_low``, ``L_fg_ci_high``, ``D_decision``, ``D_ranking``,
        ``D_total`` (``= L_fg - L_anc = D_decision + D_ranking``).
    """
    df = collect_anchored_reselection_shot_table(manager, config)
    delta_anc = df["delta_anc"].to_numpy().astype(float)
    delta_fg = df["delta_fg"].to_numpy().astype(float)
    y0 = df["y0"].to_numpy().astype(bool)
    y_star = df["y_star"].to_numpy().astype(bool)
    batch = df["batch"].to_numpy()
    shot_id = df["shot_id"].to_numpy()

    r_minus = compute_negative_anchored_rate(df)

    if abort_rates is None:
        rates = _default_abort_rate_grid(r_minus, num_points, max_multiple)
    else:
        rates = np.unique(np.clip(np.asarray(abort_rates, dtype=float), 0.0, 0.999999))

    rows: list[dict[str, Any]] = []
    for r in rates:
        accept_anc = _topk_accept_mask(delta_anc, float(r), batch, shot_id)
        accept_fg = _topk_accept_mask(delta_fg, float(r), batch, shot_id)
        denom = 1.0 - float(r)

        n_acc_anc = int(accept_anc.sum())
        n_acc_fg = int(accept_fg.sum())
        n_err_anc = int(np.sum(y0 & accept_anc))
        n_err_fg = int(np.sum(y_star & accept_fg))

        p_y0_acc_fg = float(np.mean(y0 & accept_fg))
        p_ystar_acc_fg = float(np.mean(y_star & accept_fg))
        p_y0_acc_anc = float(np.mean(y0 & accept_anc))

        l_anc = n_err_anc / n_acc_anc if n_acc_anc > 0 else np.nan
        l_fg = n_err_fg / n_acc_fg if n_acc_fg > 0 else np.nan
        l_anc_lo, l_anc_hi = (
            wilson_ci(n_err_anc, n_acc_anc, alpha=alpha) if n_acc_anc > 0 else (np.nan, np.nan)
        )
        l_fg_lo, l_fg_hi = (
            wilson_ci(n_err_fg, n_acc_fg, alpha=alpha) if n_acc_fg > 0 else (np.nan, np.nan)
        )

        d_decision = (p_ystar_acc_fg - p_y0_acc_fg) / denom if denom > 0 else np.nan
        d_ranking = (p_y0_acc_fg - p_y0_acc_anc) / denom if denom > 0 else np.nan

        rows.append(
            {
                "abort_rate": float(r),
                "r_minus": float(r_minus),
                "L_anc": l_anc,
                "L_anc_ci_low": float(l_anc_lo),
                "L_anc_ci_high": float(l_anc_hi),
                "L_fg": l_fg,
                "L_fg_ci_low": float(l_fg_lo),
                "L_fg_ci_high": float(l_fg_hi),
                "D_decision": d_decision,
                "D_ranking": d_ranking,
                "D_total": d_decision + d_ranking,
            }
        )

    return pl.DataFrame(rows).sort("abort_rate")


class AnchoredReselectionAnalyzer:
    """Plotting helper for the anchored-vs-ordinary forced-gap diagnostics."""

    # ------------------------------------------------------------------
    # Plot (a): reselection transition on negative-anchored shots
    # ------------------------------------------------------------------

    def compute_transition_table(
        self,
        manager: SimulationDataManager,
        config: AnchoredReselectionConfig,
        *,
        bins: int | Sequence[float] = 10,
        alpha: float = 0.05,
        bin_scale: BinScale = "gap_value",
        x_range: tuple[float, float] | None = None,
    ) -> pl.DataFrame:
        return compute_reselection_transition_by_forced_gap_bin(
            manager, config, bins=bins, alpha=alpha, bin_scale=bin_scale, x_range=x_range,
        )

    def compute_net_decision_effect(
        self,
        manager: SimulationDataManager,
        config: AnchoredReselectionConfig,
    ) -> dict[str, float]:
        return compute_negative_anchored_net_decision_effect(manager, config)

    def plot_reselection_transition(
        self,
        manager: SimulationDataManager,
        config: AnchoredReselectionConfig,
        ax: plt.Axes,
        *,
        bins: int | Sequence[float] = 10,
        bin_scale: BinScale = "gap_value",
        x_range: tuple[float, float] | None = None,
        category_order: Sequence[TransitionCategory] = DEFAULT_CATEGORY_ORDER,
        bar_kw: dict[str, Any] | None = None,
    ) -> plt.Axes:
        """Draw the stacked-bar transition chart for plot (a).

        Each bar corresponds to one bin within the negative-anchored subset
        ``A``; the four stacked segments are the conditional probabilities of
        the transition categories and sum to 1.

        Parameters
        ----------
        bin_scale:
            ``"gap_value"`` (default) bins by the raw ``Delta_fg`` value.
            ``"abort_rate"`` bins by the fraction of the *full* population
            that ordinary forced gap would reject at that gap value, so the
            x-axis lines up with the rejection-rate axis used by
            :meth:`plot_postselection_decomposition`.
        x_range:
            Optional ``(min, max)`` override for the binned range (units of
            ``bin_scale``); only used when ``bins`` is an integer. Shots
            outside this range are clamped into the first/last bin. ``None``
            uses the observed range within ``A``.
        """
        table = compute_reselection_transition_by_forced_gap_bin(
            manager, config, bins=bins, bin_scale=bin_scale, x_range=x_range,
        )
        bin_centers = sorted(table["bin_center"].unique().to_list())
        x = np.arange(len(bin_centers), dtype=float)
        bottom = np.zeros(len(bin_centers))

        default_bar_kw: dict[str, Any] = {"width": 0.85, "alpha": 0.9, "edgecolor": "black", "linewidth": 0.6}
        default_bar_kw.update(bar_kw or {})

        for name in category_order:
            sub = table.filter(pl.col("category") == name)
            value_map = dict(zip(sub["bin_center"].to_list(), sub["fraction"].to_list()))
            heights = np.nan_to_num(
                np.array([value_map.get(c, np.nan) for c in bin_centers], dtype=float),
                nan=0.0,
            )
            ax.bar(
                x, heights, bottom=bottom,
                label=_CATEGORY_LABELS[name], color=_CATEGORY_COLORS[name],
                **default_bar_kw,
            )
            bottom += heights

        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.3g}" for c in bin_centers], rotation=45, ha="right")
        if bin_scale == "abort_rate":
            ax.set_xlabel(
                r"Forced-gap abort rate $r=P(\Delta_{\rm fg}\le g)$ bin "
                r"(within $A=\{\Delta_{\rm anc}<0\}$)"
            )
        else:
            ax.set_xlabel(r"$\Delta_{\rm fg}$ bin (within $A=\{\Delta_{\rm anc}<0\}$)")
        ax.set_ylabel(r"$P(\mathrm{category}\mid A \cap B_j)$")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        return ax

    # ------------------------------------------------------------------
    # Plot (b): decomposition of the post-selection advantage
    # ------------------------------------------------------------------

    def compute_decomposition_curve(
        self,
        manager: SimulationDataManager,
        config: AnchoredReselectionConfig,
        *,
        abort_rates: Sequence[float] | None = None,
        num_points: int = 30,
        max_multiple: float = 2.0,
        alpha: float = 0.05,
    ) -> pl.DataFrame:
        return compute_postselection_decomposition_curve(
            manager, config,
            abort_rates=abort_rates, num_points=num_points,
            max_multiple=max_multiple, alpha=alpha,
        )

    def plot_postselection_decomposition(
        self,
        manager: SimulationDataManager,
        config: AnchoredReselectionConfig,
        ax: plt.Axes,
        *,
        abort_rates: Sequence[float] | None = None,
        num_points: int = 30,
        max_multiple: float = 2.0,
        alpha: float = 0.05,
        mark_r_minus: bool = True,
        plot_kw: dict[str, Any] | None = None,
    ) -> plt.Axes:
        """Draw ``D_decision(r)``, ``D_ranking(r)``, and ``D_total(r)`` for plot (b)."""
        df = compute_postselection_decomposition_curve(
            manager, config,
            abort_rates=abort_rates, num_points=num_points,
            max_multiple=max_multiple, alpha=alpha,
        )
        plot_kw = {"marker": "o", "markersize": 4, **(plot_kw or {})}
        r = df["abort_rate"].to_numpy()

        ax.plot(r, df["D_decision"].to_numpy(), label=r"$D_{\rm decision}(r)$", **plot_kw)
        ax.plot(r, df["D_ranking"].to_numpy(), label=r"$D_{\rm ranking}(r)$", **plot_kw)
        ax.plot(
            r, df["D_total"].to_numpy(),
            label=r"$D_{\rm total}(r) = L_{\rm fg}(r) - L_{\rm anc}(r)$",
            color="black", linestyle="--", **{**plot_kw, "marker": "s"},
        )
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")

        if mark_r_minus and df.height > 0:
            r_minus = float(df["r_minus"][0])
            ax.axvline(r_minus, color="gray", linewidth=1.0, linestyle="--", label=r"$r_-=P(\Delta_{\rm anc}<0)$")

        ax.set_xlabel(r"Matched rejection rate $r$")
        ax.set_ylabel(r"$L_{\rm fg}(r) - L_{\rm anc}(r)$ decomposition")
        ax.legend()
        return ax
