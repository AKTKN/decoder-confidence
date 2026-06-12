"""Wilson confidence interval utilities for proportion-based scatter plots.

This module is intentionally independent of the rest of the analysis package so
that it can be reused or replaced without touching the analyzer classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from statsmodels.stats.proportion import proportion_confint


@dataclass
class BinnedProportions:
    """Per-bin summary produced by :func:`bin_proportions` or :func:`bin_density`.

    Attributes
    ----------
    centers : ndarray
        Bin centre x-values.
    proportions : ndarray
        Success fraction per bin (NaN for empty bins).
    ci_low, ci_high : ndarray
        Lower / upper Wilson CI bounds (NaN for empty bins).
    counts : ndarray
        Number of successes per bin.
    totals : ndarray
        Denominator per bin (either bin count or a fixed ``n_total``).
    """

    centers: np.ndarray
    proportions: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    counts: np.ndarray
    totals: np.ndarray


def wilson_ci(
    n_success: int | np.ndarray,
    n_total: int | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson confidence interval for a proportion via statsmodels.

    Parameters
    ----------
    n_success : int or array-like
        Number of successes.
    n_total : int or array-like
        Total number of trials.
    alpha : float
        Significance level (default 0.05 → 95 % CI).

    Returns
    -------
    (ci_low, ci_high)
    """
    return proportion_confint(n_success, n_total, alpha=alpha, method="wilson")


def bin_proportions(
    x: np.ndarray,
    success: np.ndarray,
    bins: int | np.ndarray,
    n_total: int | None = None,
    alpha: float = 0.05,
    x_range: tuple[float, float] | None = None,
) -> BinnedProportions:
    """Bin x-values and compute success proportions per bin with Wilson CI.

    Parameters
    ----------
    x : array-like
        Continuous x values (e.g. gap values).
    success : array-like
        Binary outcomes (bool, 0, or 1) aligned with *x*.
    bins : int or array-like
        Number of uniform bins **or** explicit bin edges.
    n_total : int | None
        Fixed denominator applied to every bin.  When ``None`` each bin uses
        its own observation count as the denominator.  Pass ``len(x)`` to get
        per-bin *density* (fraction of total shots).
    alpha : float
        Significance level for the CI (default 0.05 → 95 % CI).
    x_range : (float, float) | None
        ``(min, max)`` range for uniform bins.  Defaults to
        ``(x.min(), x.max())``.

    Returns
    -------
    BinnedProportions
    """
    x = np.asarray(x, dtype=float)
    success = np.asarray(success, dtype=float)

    if x.size == 0:
        empty: np.ndarray = np.array([], dtype=float)
        return BinnedProportions(empty, empty, empty, empty,
                                 np.array([], dtype=int), np.array([], dtype=int))

    lo, hi = x_range if x_range is not None else (float(x.min()), float(x.max()))

    if isinstance(bins, int):
        edges = np.linspace(lo, hi, bins + 1)
        n_bins = bins
    else:
        edges = np.asarray(bins, dtype=float)
        n_bins = len(edges) - 1

    centers = 0.5 * (edges[:-1] + edges[1:])

    # np.digitize returns 1-based indices; shift to 0-based and clamp
    indices = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    counts = np.zeros(n_bins, dtype=int)
    bin_totals = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = indices == i
        bin_totals[i] = int(mask.sum())
        counts[i] = int(success[mask].sum())

    denominators = (
        np.full(n_bins, n_total, dtype=int) if n_total is not None else bin_totals
    )
    proportions = np.where(denominators > 0, counts / denominators, np.nan)

    ci_low = np.full(n_bins, np.nan)
    ci_high = np.full(n_bins, np.nan)
    valid = denominators > 0
    if valid.any():
        low, high = wilson_ci(counts[valid], denominators[valid], alpha=alpha)
        ci_low[valid] = low
        ci_high[valid] = high

    return BinnedProportions(
        centers=centers,
        proportions=proportions,
        ci_low=ci_low,
        ci_high=ci_high,
        counts=counts,
        totals=denominators,
    )


def bin_density(
    x: np.ndarray,
    bins: int | np.ndarray,
    alpha: float = 0.05,
    x_range: tuple[float, float] | None = None,
) -> BinnedProportions:
    """Bin x-values and compute the fraction of total observations per bin.

    Equivalent to calling :func:`bin_proportions` with ``success`` set to all
    ones and ``n_total`` set to ``len(x)``.  Each bin's y-value is therefore
    the fraction of shots that land in that bin, with Wilson CI computed
    against the total shot count.
    """
    x = np.asarray(x, dtype=float)
    return bin_proportions(
        x=x,
        success=np.ones(len(x)),
        bins=bins,
        n_total=len(x),
        alpha=alpha,
        x_range=x_range,
    )


def value_proportions(
    x: np.ndarray,
    success: np.ndarray,
    alpha: float = 0.05,
) -> BinnedProportions:
    """Compute success proportions per *unique* x-value with Wilson CI.

    Unlike :func:`bin_proportions`, no binning is performed: each distinct
    value of *x* becomes its own group, with ``proportions[i]`` the fraction
    of ``success`` among observations where ``x == centers[i]``.

    Parameters
    ----------
    x : array-like
        Values to group by (e.g. gap values).
    success : array-like
        Binary outcomes (bool, 0, or 1) aligned with *x*.
    alpha : float
        Significance level for the CI (default 0.05 -> 95% CI).

    Returns
    -------
    BinnedProportions
    """
    x = np.asarray(x, dtype=float)
    success = np.asarray(success, dtype=float)

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
        counts[i] = int(success[mask].sum())

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


def shade_ci(
    ax: Axes,
    x: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    color,
    alpha: float = 0.2,
) -> None:
    """Shade the CI band on *ax* using :func:`matplotlib.axes.Axes.fill_between`.

    NaN values in any of the three arrays are excluded automatically so that
    empty bins do not produce artefacts.
    """
    valid = ~(np.isnan(x) | np.isnan(ci_low) | np.isnan(ci_high))
    if valid.any():
        ax.fill_between(x[valid], ci_low[valid], ci_high[valid],
                        alpha=alpha, color=color)
