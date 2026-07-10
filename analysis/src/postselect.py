"""Post-selection performance plots.

For each confidence metric, sweep an abort threshold to trace out the trade-off
between abort rate (fraction of shots discarded) and post-selected logical
error rate (LER measured on the accepted subset). Wilson confidence intervals
are shaded around each curve.

Direction depends on the metric:

* ``logical_gap``, ``linearize_logicalgap``, ``reweighted_linearized_gap``, ``forced_gap_ml``: higher = more
  confident, so low-value shots are aborted preferentially (``direction="high"``).
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

from analysis.src.case_histogram import _CIRCUIT_PARAM_KEYS
from analysis.src.config import PlotConfig
from analysis.src.confidence import shade_ci, wilson_ci
from analysis.src.data_manager import SimulationDataManager


HIGH_CONFIDENCE_METRICS: frozenset[str] = frozenset(
    {
        "logical_gap",
        "linearize_logicalgap",
        "reweighted_linearized_gap",
        "forced_gap_ml",
    }
)
LOW_CONFIDENCE_METRICS: frozenset[str] = frozenset({"cluster_llr"})
BOOLEAN_METRICS: frozenset[str] = frozenset({"ar-pec", "ar-lec", "argument_reweighting"})


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
    mark_zero_gap:
        When ``True`` and ``metric_name == "linearize_logicalgap"``, overlay a
        star marker at the operating point obtained by discarding all shots
        with ``linearize_logicalgap <= 0``.
    separate_unconverged:
        Only supported for ``linearize_logicalgap`` and ``forced_gap_ml``
        (RELAY-BP's ``get_detail_stat=True`` directories). When ``True``,
        shots whose first-stage (baseline) decode failed to converge
        (``decoder_stat``'s ``baseline_iteration`` is NaN) are treated as a
        forced-error, always-aborted category rather than being swept over
        with the normal gap threshold — their placeholder gap value does not
        reflect real confidence. The curve then starts from the operating
        point that aborts exactly these shots and accepts every converged
        shot, after which it proceeds through finite gap thresholds as
        usual. That starting point's post-selected LER is also drawn as a
        horizontal dotted reference line, in the same color as the curve.
    exact_gap_ranking:
        Only supported for ``linearize_logicalgap`` and ``forced_gap_ml``.
        When ``True``, per-shot logical-error labels (``is_logical_error``)
        still come from ``metric_name``'s own decoder/metric directory, but
        the value used to rank shots for post-selection (the abort
        threshold sweep, ``mark_zero_gap``, ``separate_unconverged``, etc.)
        is replaced by the shot-matched ``logical_gap`` value (typically
        from the exact ``ILP`` decoder) instead of ``metric_name``'s own gap
        value. The two datasets are joined on ``shot_id`` (plus shared
        circuit-parameter columns). Raises ``FileNotFoundError`` if no
        ``logical_gap`` data is found, or ``ValueError`` if the join leaves
        no matching shots. A spec with ``exact_gap_ranking=True`` and one
        with ``exact_gap_ranking=False`` for the same ``metric_name`` can be
        passed together to :meth:`PostSelectionPlotter.plot` to draw both on
        the same axes; unless ``label_prefix`` is set, ``" (exact gap
        ranking)"`` is appended to the legend label to tell them apart.
    exact_gap_decoder_names:
        Only used when ``exact_gap_ranking=True``. Decoder name(s) to load
        the ``logical_gap`` values from (e.g. ``["ILP"]``). ``None``
        includes every decoder found on disk for ``logical_gap``.
    """

    metric_name: str
    decoder_names: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    batch_indices: Optional[List[int]] = None
    direction: Optional[str] = None
    label_prefix: Optional[str] = None
    round_digit: Optional[int] = None
    threshold_mode: str = "continuous"
    grid_scale: str = "linear"
    use_linearize: bool = False
    mark_zero_gap: bool = False
    get_detail_stat: Optional[bool] = None
    random_split: Optional[bool] = None
    n_splits: Optional[int] = None
    split_seed: Optional[int] = None
    separate_unconverged: bool = False
    exact_gap_ranking: bool = False
    exact_gap_decoder_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        split_metric = self.metric_name in {"linearize_logicalgap", "forced_gap_ml"}
        if self.random_split is not None or self.n_splits is not None:
            if not split_metric:
                raise ValueError(
                    "random_split/n_splits options are only supported for "
                    "linearize_logicalgap and forced_gap_ml"
                )
            if self.n_splits is not None and self.random_split is None:
                raise ValueError("n_splits requires random_split to be specified")
            # random_split=True needs n_splits to pick out a specific split
            # count; random_split=False directories may or may not encode
            # n_splits in their name (e.g. "...,random_split=False" with no
            # n_splits key at all), so n_splits stays optional in that case.
            if self.random_split is True and self.n_splits is None:
                raise ValueError("random_split=True requires n_splits to be specified")
        if self.separate_unconverged and not split_metric:
            raise ValueError(
                "separate_unconverged is only supported for "
                "linearize_logicalgap and forced_gap_ml"
            )
        if self.exact_gap_ranking and not split_metric:
            raise ValueError(
                "exact_gap_ranking is only supported for "
                "linearize_logicalgap and forced_gap_ml"
            )
        if self.exact_gap_decoder_names is not None and not self.exact_gap_ranking:
            raise ValueError(
                "exact_gap_decoder_names requires exact_gap_ranking=True"
            )

    def decoder_query_options(self) -> dict[str, Any]:
        """Return decoder-directory filters for this split configuration.

        ``random_split=None`` means the ordinary, non-split metric directory.
        """
        if self.metric_name not in {"linearize_logicalgap", "forced_gap_ml"}:
            return {}

        detail_filter: dict[str, Any] = {}
        if self.get_detail_stat is not None:
            detail_filter["get_detail_stat"] = bool(self.get_detail_stat)

        if self.random_split is None:
            if detail_filter:
                options: dict[str, Any] = {
                    "decoder_filters": detail_filter,
                    "decoder_exclude_keys": ["random_split", "randon_split", "n_splits"],
                }
            else:
                options = {
                    "decoder_exclude_keys": ["random_split", "randon_split", "n_splits", "get_detail_stat"]
                }
        else:
            filters: dict[str, Any] = {
                **detail_filter,
                "random_split": bool(self.random_split),
            }
            exclude_keys: list[str] = []
            if self.n_splits is not None:
                filters["n_splits"] = int(self.n_splits)
            else:
                exclude_keys.append("n_splits")
            if self.split_seed is not None:
                filters["split_seed"] = int(self.split_seed)
            elif exclude_keys:
                exclude_keys.append("split_seed")

            options = {"decoder_filters": filters}
            if exclude_keys:
                options["decoder_exclude_keys"] = exclude_keys

        if self.separate_unconverged:
            options["extra_stat_files"] = [
                {"file_stem": "decoder_stat", "columns": ["baseline_iteration"]}
            ]
        return options


@dataclass
class PostSelectCurve:
    """Per-point post-selection statistics."""

    abort_rates: np.ndarray
    post_lers: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    accepted: np.ndarray
    logical_errors: np.ndarray


def postselect_point_at_threshold(
    values: np.ndarray,
    is_error: np.ndarray,
    threshold: float,
    direction: str,
    alpha: float = 0.05,
    *,
    inclusive: bool = True,
) -> PostSelectCurve:
    """Return the single operating point defined by a fixed threshold."""
    if direction not in {"high", "low"}:
        raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")

    values = np.asarray(values, dtype=float)
    is_error = np.asarray(is_error, dtype=bool)
    n_total = values.size

    if n_total == 0:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        return PostSelectCurve(empty_f, empty_f, empty_f, empty_f, empty_i, empty_i)

    if direction == "high":
        mask = values >= threshold if inclusive else values > threshold
    else:
        mask = values <= threshold if inclusive else values < threshold

    n_acc = int(mask.sum())
    abort_rate = 1.0 - n_acc / n_total

    if n_acc == 0:
        nan = np.array([np.nan], dtype=float)
        return PostSelectCurve(
            abort_rates=np.array([abort_rate], dtype=float),
            post_lers=nan,
            ci_low=nan.copy(),
            ci_high=nan.copy(),
            accepted=np.array([0], dtype=int),
            logical_errors=np.array([0], dtype=int),
        )

    k_err = int(is_error[mask].sum())
    post_ler = k_err / n_acc
    lo, hi = wilson_ci(k_err, n_acc, alpha=alpha)
    return PostSelectCurve(
        abort_rates=np.array([abort_rate], dtype=float),
        post_lers=np.array([post_ler], dtype=float),
        ci_low=np.array([float(lo)], dtype=float),
        ci_high=np.array([float(hi)], dtype=float),
        accepted=np.array([n_acc], dtype=int),
        logical_errors=np.array([k_err], dtype=int),
    )


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
        return PostSelectCurve(empty_f, empty_f, empty_f, empty_f, empty_i, empty_i)

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
    logical_errors = np.zeros(num_points, dtype=int)

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
        logical_errors[i] = k_err
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
        logical_errors=logical_errors[order],
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
        return PostSelectCurve(empty_f, empty_f, empty_f, empty_f, empty_i, empty_i)

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
        rows.append((abort, post_ler, float(lo), float(hi), n_acc, k_err))

    if not rows:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        return PostSelectCurve(empty_f, empty_f, empty_f, empty_f, empty_i, empty_i)

    rows.sort(key=lambda r: r[0])
    abort_arr, ler_arr, lo_arr, hi_arr, acc_arr, err_arr = zip(*rows)
    return PostSelectCurve(
        abort_rates=np.array(abort_arr, dtype=float),
        post_lers=np.array(ler_arr, dtype=float),
        ci_low=np.array(lo_arr, dtype=float),
        ci_high=np.array(hi_arr, dtype=float),
        accepted=np.array(acc_arr, dtype=int),
        logical_errors=np.array(err_arr, dtype=int),
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
            rows.append((b_val, abort, np.nan, np.nan, np.nan, 0, 0))
            continue

        k_err = int(is_err[accept].sum())
        post_ler = k_err / n_acc
        lo, hi = wilson_ci(k_err, n_acc, alpha=alpha)
        rows.append((b_val, abort, post_ler, float(lo), float(hi), n_acc, k_err))

    if not rows:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        return PostSelectCurve(empty_f, empty_f, empty_f, empty_f, empty_i, empty_i)

    rows.sort(key=lambda r: r[1])
    _, abort, ler, lo, hi, acc, err = zip(*rows)
    return PostSelectCurve(
        abort_rates=np.array(abort, dtype=float),
        post_lers=np.array(ler, dtype=float),
        ci_low=np.array(lo, dtype=float),
        ci_high=np.array(hi, dtype=float),
        accepted=np.array(acc, dtype=int),
        logical_errors=np.array(err, dtype=int),
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
            extra_options={
                "use_linearize": spec.use_linearize,
                **spec.decoder_query_options(),
            },
        )
        lf = self.manager.query(plot_config)
        if spec.exact_gap_ranking:
            lf = self._join_exact_gap_ranking(lf, spec)

        needed = [spec.metric_name, "is_logical_error"] + list(spec.group_by)
        if direction == "boolean":
            needed.append("b")
        if spec.separate_unconverged:
            needed.append("baseline_iteration")
        schema_names = set(lf.collect_schema().names())
        existing = [c for c in dict.fromkeys(needed) if c in schema_names]
        df = lf.select(existing).collect()

        if spec.exact_gap_ranking and df.height == 0:
            raise ValueError(
                "exact_gap_ranking=True: no shots remain after joining "
                f"'{spec.metric_name}' with 'logical_gap' data on shot_id "
                f"(filters={spec.filters}). Check that logical_gap data exists "
                "for the same shots."
            )

        prefix = f"{spec.label_prefix} " if spec.label_prefix else ""
        suffix = " (exact gap ranking)" if spec.exact_gap_ranking else ""

        if spec.group_by:
            partitions: dict[tuple, pl.DataFrame] = df.partition_by(
                spec.group_by, as_dict=True
            )
            for key_vals in sorted(partitions):
                part_df = partitions[key_vals]
                key_tuple = key_vals if isinstance(key_vals, tuple) else (key_vals,)
                group_label = _format_label(list(zip(spec.group_by, key_tuple)))
                label = f"{prefix}{spec.metric_name} | {group_label}{suffix}"
                self._compute_and_plot(
                    part_df, spec, direction, ax,
                    num_points, alpha, shade_alpha, plot_kw, label, reduction_rate,
                )
        else:
            label = f"{prefix}{spec.metric_name}{suffix}"
            self._compute_and_plot(
                df, spec, direction, ax,
                num_points, alpha, shade_alpha, plot_kw, label, reduction_rate,
            )

    def _join_exact_gap_ranking(
        self, lf: pl.LazyFrame, spec: PostSelectSpec
    ) -> pl.LazyFrame:
        """Replace *lf*'s ``spec.metric_name`` column with shot-matched ``logical_gap``.

        ``is_logical_error`` and every other column of *lf* are left as-is —
        they still come from ``spec.metric_name``'s own decoder/metric
        directory. Only the ranking value itself is overwritten so
        downstream logic (threshold sweeps, ``mark_zero_gap``,
        ``separate_unconverged``, ...) can keep referring to
        ``spec.metric_name`` unchanged.
        """
        logical_gap_config = PlotConfig(
            metric_name="logical_gap",
            decoder_names=spec.exact_gap_decoder_names,
            filters=spec.filters,
            batch_indices=spec.batch_indices,
        )
        try:
            lg_lf = self.manager.query(logical_gap_config)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "exact_gap_ranking=True requires 'logical_gap' data, but none "
                f"was found for filters={spec.filters}: {exc}"
            ) from exc

        schema_main = set(lf.collect_schema().names())
        schema_lg = set(lg_lf.collect_schema().names())
        join_keys = ["shot_id"] + [
            k for k in _CIRCUIT_PARAM_KEYS if k in schema_main and k in schema_lg
        ]

        lg_cols = list(
            dict.fromkeys([k for k in join_keys if k in schema_lg] + ["logical_gap"])
        )
        lg_lf = lg_lf.select(lg_cols).rename({"logical_gap": "__exact_gap_ranking_value"})

        joined = lf.join(lg_lf, on=join_keys, how="inner")
        return joined.drop(spec.metric_name).rename(
            {"__exact_gap_ranking_value": spec.metric_name}
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
        baseline_ler: Optional[float] = None
        if direction == "boolean":
            curve = postselect_curve_ar(df, spec.metric_name, alpha=alpha)
            original_ler = df["is_logical_error"].mean()
        elif spec.separate_unconverged:
            sub = df.drop_nulls([spec.metric_name, "is_logical_error"])
            curve, original_ler, baseline_ler = self._compute_curve_separate_unconverged(
                sub, spec, direction, num_points, alpha,
            )
        else:
            sub = df.drop_nulls([spec.metric_name, "is_logical_error"])
            values = sub[spec.metric_name].to_numpy().astype(float)

            if spec.round_digit is not None:
                values = np.round(values, spec.round_digit)

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

        has_logical_error = curve.logical_errors > 0
        y = np.where(has_logical_error, y, np.nan)
        ci_lo = np.where(has_logical_error, ci_lo, np.nan)
        ci_hi = np.where(has_logical_error, ci_hi, np.nan)

        valid = ~(np.isnan(y) | np.isnan(curve.abort_rates))
        if not valid.any():
            return
        use_markers = direction == "boolean" or spec.threshold_mode == "unique"
        if use_markers:
            line, = ax.plot(
                curve.abort_rates, y,
                marker="x", linestyle="-", label=label, **plot_kw,
            )
        else:
            line, = ax.plot(
                curve.abort_rates, y,
                label=label, **plot_kw,
            )
        shade_ci(
            ax, curve.abort_rates, ci_lo, ci_hi,
            color=line.get_color(), alpha=shade_alpha,
        )
        if baseline_ler is not None:
            y_base = baseline_ler
            if reduction_rate:
                y_base = y_base / original_ler if original_ler and original_ler > 0 else None
            if y_base is not None and not np.isnan(y_base):
                ax.axhline(
                    y_base, color=line.get_color(), linestyle=":",
                    linewidth=1.2, alpha=0.8, label="_nolegend_",
                )
        self._plot_zero_gap_marker(
            ax=ax,
            df=df,
            spec=spec,
            direction=direction,
            alpha=alpha,
            reduction_rate=reduction_rate,
            original_ler=original_ler,
            color=line.get_color(),
        )

    def _compute_curve_separate_unconverged(
        self,
        sub: pl.DataFrame,
        spec: PostSelectSpec,
        direction: str,
        num_points: int,
        alpha: float,
    ) -> tuple[PostSelectCurve, float, Optional[float]]:
        """Post-selection curve that always aborts stage-1-unconverged shots first.

        Shots whose baseline (first-stage) decode failed to converge get a
        placeholder gap value (e.g. 0 for ``forced_gap_ml``, ``-inf`` for
        ``linearize_logicalgap``) that does not reflect real confidence, so
        they are pulled out of the normal threshold sweep entirely. The
        sweep runs only over the remaining (converged) shots, but abort
        rates stay expressed relative to the full shot count, so the curve's
        first point is exactly "abort every unconverged shot, accept every
        converged shot" and later points proceed through finite gap values
        as usual.
        """
        n_total = sub.height
        unconverged_mask = sub["baseline_iteration"].is_nan()
        converged = sub.filter(~unconverged_mask)
        unconverged = sub.filter(unconverged_mask)

        values = converged[spec.metric_name].to_numpy().astype(float)
        if spec.round_digit is not None:
            values = np.round(values, spec.round_digit)
        is_error = converged["is_logical_error"].to_numpy().astype(bool)

        if spec.threshold_mode == "unique":
            curve = postselect_curve_unique_thresholds(values, is_error, direction, alpha=alpha)
        else:
            curve = postselect_curve_continuous(
                values, is_error, direction,
                num_points=num_points, alpha=alpha, grid_scale=spec.grid_scale,
            )

        if n_total > 0 and curve.abort_rates.size > 0:
            curve.abort_rates = 1.0 - curve.accepted / n_total

        original_ler = float(sub["is_logical_error"].to_numpy().mean()) if n_total > 0 else np.nan

        baseline_ler: Optional[float] = None
        if unconverged.height > 0 and values.size > 0:
            baseline = postselect_point_at_threshold(
                values, is_error, threshold=float(np.min(values)),
                direction=direction, alpha=alpha, inclusive=True,
            )
            if baseline.accepted.size > 0 and baseline.accepted[0] > 0:
                baseline_ler = float(baseline.post_lers[0])

        return curve, original_ler, baseline_ler

    def _plot_zero_gap_marker(
        self,
        ax: plt.Axes,
        df: pl.DataFrame,
        spec: PostSelectSpec,
        direction: str,
        alpha: float,
        reduction_rate: bool,
        original_ler: float,
        color: Any,
    ) -> None:
        if not spec.mark_zero_gap or spec.metric_name != "linearize_logicalgap":
            return
        if direction == "boolean":
            return

        sub = df.drop_nulls([spec.metric_name, "is_logical_error"])
        if sub.height == 0:
            return

        values = sub[spec.metric_name].to_numpy().astype(float)
        is_error = sub["is_logical_error"].to_numpy().astype(bool)
        zero_gap = postselect_point_at_threshold(
            values,
            is_error,
            threshold=0.0,
            direction=direction,
            alpha=alpha,
            inclusive=False if direction == "high" else True,
        )
        if (
            zero_gap.abort_rates.size == 0
            or np.isnan(zero_gap.post_lers[0])
            or zero_gap.logical_errors[0] <= 0
        ):
            return

        y = float(zero_gap.post_lers[0])
        if reduction_rate:
            if not original_ler or original_ler <= 0:
                return
            y /= original_ler

        ax.plot(
            [float(zero_gap.abort_rates[0])],
            [y],
            marker="*",
            linestyle="None",
            color=color,
            markersize=12,
            label="_nolegend_",
        )
