from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis.src.confidence import wilson_ci
from analysis.src.data_manager import SimulationDataManager
from analysis.src.stage_gap_difference import (
    Stage1GapDifferenceConfig,
    load_stage1_gap_difference_lazy,
)

ScoreName = Literal[
    "exact_gap",
    "forced_gap",
    "anchored_gap",
    "forced_gap_override_penalty",
    "hybrid_min_forced_anchor",
]

RankingMetricName = Literal[
    "forced_only_fraction",
    "exact_only_fraction",
    "jaccard_distance",
    "symmetric_difference_fraction",
    "forced_only_among_forced_accepted",
    "exact_only_among_exact_accepted",
]

ThresholdRejectionSet = Literal[
    "both_rejected",
    "overconfident",
    "underconfident",
]

_SCORE_LABELS: dict[str, str] = {
    "exact_gap": r"$\Delta_{\rm ex}$",
    "forced_gap": r"$\Delta_{\rm fg}$",
    "anchored_gap": r"$\Delta_{\rm anchor}$",
    "forced_gap_override_penalty": r"$\Delta_{\rm fg}$ + override penalty",
    "hybrid_min_forced_anchor": r"$\min(\Delta_{\rm fg},\Delta_{\rm anchor})$",
}

_RANKING_METRIC_LABELS: dict[str, str] = {
    "forced_only_fraction": "Forced-only accepted fraction",
    "exact_only_fraction": "Exact-only accepted fraction",
    "jaccard_distance": "Jaccard distance",
    "symmetric_difference_fraction": "Symmetric difference fraction",
    "forced_only_among_forced_accepted": "Forced-only / forced accepted",
    "exact_only_among_exact_accepted": "Exact-only / exact accepted",
}

_THRESHOLD_SET_LABELS: dict[str, str] = {
    "both_rejected": "Both rejected",
    "overconfident": "Overconfident",
    "underconfident": "Underconfident",
}

_THRESHOLD_CATEGORY_LABELS: dict[str, str] = {
    "stage1_same_weight": "stage1 same weight",
    "stage1_different_weight_override": "stage1 different + override",
    "stage1_different_weight_no_override": "stage1 different + no override",
}


@dataclass
class RankingAnalysisConfig:
    """Configuration for ranking/post-selection analyses on detailed gap data."""

    detail_config: Stage1GapDifferenceConfig
    abort_rates: Sequence[float] = field(
        default_factory=lambda: np.linspace(0.0, 0.9, 19).tolist()
    )
    scores: Sequence[ScoreName] = (
        "exact_gap",
        "forced_gap",
        "anchored_gap",
        "forced_gap_override_penalty",
    )
    override_penalty_strength: float = 1.0
    alpha: float = 0.05


def _validate_abort_rates(abort_rates: Sequence[float]) -> np.ndarray:
    arr = np.asarray(abort_rates, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("abort_rates must be a non-empty 1D sequence")
    if np.any((arr < 0.0) | (arr >= 1.0)):
        raise ValueError("abort_rates must satisfy 0 <= r < 1")
    return arr


def _score_label(score: str) -> str:
    return _SCORE_LABELS.get(score, score)


def _ranking_metric_label(metric: str) -> str:
    return _RANKING_METRIC_LABELS.get(metric, metric)


def _threshold_set_label(name: str) -> str:
    return _THRESHOLD_SET_LABELS.get(name, name)


def _threshold_category_label(name: str) -> str:
    return _THRESHOLD_CATEGORY_LABELS.get(name, name)


def _ranking_lazy(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
) -> pl.LazyFrame:
    lf = load_stage1_gap_difference_lazy(manager.result_dir_root, config.detail_config)
    override_amount = (
        pl.col("forced_stage1_weight") - pl.col("forced_stage2_weight")
    ).clip(0.0, None)
    anchored_gap = pl.col("forced_stage2_weight") - pl.col("forced_stage1_weight")
    return lf.with_columns(
        [
            pl.col("logical_gap").alias("score_exact_gap"),
            pl.col("forced_gap_ml").alias("score_forced_gap"),
            anchored_gap.alias("score_anchored_gap"),
            (
                pl.col("forced_gap_ml")
                - float(config.override_penalty_strength) * override_amount
            ).alias("score_forced_gap_override_penalty"),
            pl.min_horizontal(pl.col("forced_gap_ml"), anchored_gap).alias(
                "score_hybrid_min_forced_anchor"
            ),
            override_amount.gt(0.0).alias("forced_override"),
            (~pl.col("stage1_weight_same")).alias("stage1_weight_different"),
            (pl.col("forced_gap_ml") - pl.col("logical_gap")).alias("margin"),
            anchored_gap.alias("anchored_gap"),
        ]
    )


def collect_ranking_shot_table(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
) -> pl.DataFrame:
    """Return the per-shot joined table with derived ranking score columns."""
    return _ranking_lazy(manager, config).collect()


def _topk_accept_mask(
    score: np.ndarray,
    abort_rate: float,
    batch: np.ndarray,
    shot_id: np.ndarray,
) -> np.ndarray:
    n = score.size
    n_accept = int(round((1.0 - float(abort_rate)) * n))
    n_accept = max(0, min(n, n_accept))
    mask = np.zeros(n, dtype=bool)
    if n_accept == 0:
        return mask

    clean_score = np.asarray(score, dtype=float)
    clean_score = np.where(np.isfinite(clean_score), clean_score, -np.inf)
    order = np.lexsort((shot_id, batch, -clean_score))
    mask[order[:n_accept]] = True
    return mask


def _np_col(df: pl.DataFrame, name: str) -> np.ndarray:
    return df.get_column(name).to_numpy()


def _score_column(score: str) -> str:
    return f"score_{score}"


def compute_ranking_degradation(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    *,
    exact_score: ScoreName = "exact_gap",
    comparison_score: ScoreName = "forced_gap",
) -> pl.DataFrame:
    """Compare exact-score and comparison-score accepted sets over abort rates."""
    abort_rates = _validate_abort_rates(config.abort_rates)
    df = collect_ranking_shot_table(manager, config)
    batch = _np_col(df, "batch")
    shot_id = _np_col(df, "shot_id")
    exact = _np_col(df, _score_column(exact_score))
    comparison = _np_col(df, _score_column(comparison_score))
    n_total = df.height

    rows: list[dict[str, Any]] = []
    for r in abort_rates:
        exact_acc = _topk_accept_mask(exact, r, batch, shot_id)
        comp_acc = _topk_accept_mask(comparison, r, batch, shot_id)

        both_accept = int(np.logical_and(exact_acc, comp_acc).sum())
        exact_only = int(np.logical_and(exact_acc, ~comp_acc).sum())
        forced_only = int(np.logical_and(~exact_acc, comp_acc).sum())
        both_reject = int(np.logical_and(~exact_acc, ~comp_acc).sum())
        union = both_accept + exact_only + forced_only
        sym_diff = exact_only + forced_only
        exact_accept = both_accept + exact_only
        comp_accept = both_accept + forced_only

        rows.append(
            {
                "abort_rate": float(r),
                "exact_score": exact_score,
                "comparison_score": comparison_score,
                "n_total": n_total,
                "n_accepted": exact_accept,
                "both_accept": both_accept,
                "exact_only": exact_only,
                "forced_only": forced_only,
                "both_reject": both_reject,
                "forced_only_fraction": forced_only / n_total,
                "exact_only_fraction": exact_only / n_total,
                "symmetric_difference_fraction": sym_diff / n_total,
                "jaccard_distance": sym_diff / union if union > 0 else 0.0,
                "forced_only_among_forced_accepted": (
                    forced_only / comp_accept if comp_accept > 0 else np.nan
                ),
                "exact_only_among_exact_accepted": (
                    exact_only / exact_accept if exact_accept > 0 else np.nan
                ),
            }
        )
    return pl.DataFrame(rows)


def compute_postselection_curves(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    *,
    error_source: Literal["score_default", "logical", "forced"] = "score_default",
) -> pl.DataFrame:
    """Compute post-selected LER curves for configured confidence scores."""
    abort_rates = _validate_abort_rates(config.abort_rates)
    df = collect_ranking_shot_table(manager, config)
    batch = _np_col(df, "batch")
    shot_id = _np_col(df, "shot_id")
    logical_err = _np_col(df, "logical_is_logical_error").astype(bool)
    forced_err = _np_col(df, "forced_is_logical_error").astype(bool)
    rows: list[dict[str, Any]] = []

    for score_name in config.scores:
        score = _np_col(df, _score_column(score_name))
        if error_source == "logical":
            errors = logical_err
            err_label = "logical"
        elif error_source == "forced":
            errors = forced_err
            err_label = "forced"
        elif score_name == "exact_gap":
            errors = logical_err
            err_label = "logical"
        else:
            errors = forced_err
            err_label = "forced"

        for r in abort_rates:
            accepted = _topk_accept_mask(score, r, batch, shot_id)
            n_acc = int(accepted.sum())
            if n_acc == 0:
                ler = np.nan
                ci_low = np.nan
                ci_high = np.nan
                n_err = 0
            else:
                n_err = int(errors[accepted].sum())
                ler = n_err / n_acc
                ci_low, ci_high = wilson_ci(n_err, n_acc, alpha=config.alpha)

            rows.append(
                {
                    "score": score_name,
                    "score_label": _score_label(score_name),
                    "error_source": err_label,
                    "abort_rate": float(r),
                    "acceptance_rate": n_acc / df.height,
                    "n_accepted": n_acc,
                    "n_logical_error": n_err,
                    "post_ler": float(ler),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                }
            )
    return pl.DataFrame(rows)


def _forced_only_mask(
    df: pl.DataFrame,
    abort_rate: float,
    exact_score: ScoreName,
    comparison_score: ScoreName,
) -> np.ndarray:
    batch = _np_col(df, "batch")
    shot_id = _np_col(df, "shot_id")
    exact = _np_col(df, _score_column(exact_score))
    comparison = _np_col(df, _score_column(comparison_score))
    exact_acc = _topk_accept_mask(exact, abort_rate, batch, shot_id)
    comp_acc = _topk_accept_mask(comparison, abort_rate, batch, shot_id)
    return np.logical_and(~exact_acc, comp_acc)


def compute_false_safe_decomposition(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    *,
    abort_rates: Sequence[float] | None = None,
    exact_score: ScoreName = "exact_gap",
    comparison_score: ScoreName = "forced_gap",
    group_by: Sequence[str] = (
        "stage1_weight_same",
        "forced_override",
        "forced_stage1_obs_flip",
        "forced_stage2_obs_flip",
        "forced_stage2_2ndbest_obs_flip",
    ),
) -> pl.DataFrame:
    """Aggregate diagnostics inside shots accepted only by the comparison score."""
    rates = _validate_abort_rates(abort_rates or config.abort_rates)
    df = collect_ranking_shot_table(manager, config)
    frames: list[pl.DataFrame] = []

    for r in rates:
        mask = _forced_only_mask(df, float(r), exact_score, comparison_score)
        sub = df.filter(pl.Series(mask))
        if sub.height == 0:
            continue
        grouped = (
            sub.group_by(list(group_by))
            .agg(
                [
                    pl.len().alias("n"),
                    pl.col("forced_is_logical_error").cast(pl.Float64).mean().alias(
                        "logical_error_rate"
                    ),
                    pl.col("logical_gap").mean().alias("mean_exact_gap"),
                    pl.col("forced_gap_ml").mean().alias("mean_forced_gap"),
                    pl.col("margin").mean().alias("mean_margin"),
                    pl.col("anchored_gap").mean().alias("mean_anchored_gap"),
                ]
            )
            .with_columns(
                [
                    pl.lit(float(r)).alias("abort_rate"),
                    (pl.col("n") / sub.height).alias("fraction_within_forced_only"),
                    pl.lit(sub.height).alias("n_forced_only"),
                ]
            )
        )
        frames.append(grouped)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal").sort(["abort_rate", "n"], descending=[False, True])


def compute_false_safe_indicator_fractions(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    *,
    abort_rates: Sequence[float],
    exact_score: ScoreName = "exact_gap",
    comparison_score: ScoreName = "forced_gap",
) -> pl.DataFrame:
    """Compute independent binary indicator fractions in forced-only accepted shots."""
    rates = _validate_abort_rates(abort_rates)
    df = collect_ranking_shot_table(manager, config)
    indicators: dict[str, np.ndarray] = {
        "stage-1 weight different": _np_col(df, "stage1_weight_different").astype(bool),
        "override": _np_col(df, "forced_override").astype(bool),
        "stage1 logical error": _np_col(df, "forced_stage1_obs_flip").astype(bool),
        "stage2 logical error": _np_col(df, "forced_stage2_obs_flip").astype(bool),
        "stage2 second-best logical error": _np_col(
            df, "forced_stage2_2ndbest_obs_flip"
        ).astype(bool),
    }
    rows: list[dict[str, Any]] = []
    for r in rates:
        mask = _forced_only_mask(df, float(r), exact_score, comparison_score)
        n = int(mask.sum())
        for label, values in indicators.items():
            rows.append(
                {
                    "abort_rate": float(r),
                    "indicator": label,
                    "fraction": float(values[mask].mean()) if n > 0 else np.nan,
                    "n_forced_only": n,
                }
            )
    return pl.DataFrame(rows)


def compute_rank_correlations(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    *,
    score_pairs: Sequence[tuple[ScoreName, ScoreName]] = (("exact_gap", "forced_gap"),),
) -> pl.DataFrame:
    """Compute Spearman and Kendall correlations for score rankings."""
    from scipy.stats import kendalltau, spearmanr

    df = collect_ranking_shot_table(manager, config)
    rows: list[dict[str, Any]] = []
    for left, right in score_pairs:
        x = _np_col(df, _score_column(left))
        y = _np_col(df, _score_column(right))
        finite = np.isfinite(x) & np.isfinite(y)
        spearman = spearmanr(x[finite], y[finite])
        kendall = kendalltau(x[finite], y[finite])
        rows.append(
            {
                "left_score": left,
                "right_score": right,
                "n": int(finite.sum()),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_p": float(kendall.pvalue),
            }
        )
    return pl.DataFrame(rows)


def _logical_gap_thresholds(
    df: pl.DataFrame,
    thresholds: Sequence[float] | None,
) -> np.ndarray:
    if thresholds is not None:
        arr = np.asarray(thresholds, dtype=float)
    else:
        arr = np.asarray(df["logical_gap_value"].unique().sort().to_list(), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("thresholds must be a non-empty 1D sequence")
    return arr


def _threshold_rejection_masks(
    df: pl.DataFrame,
    threshold: float,
) -> dict[str, np.ndarray]:
    logical = _np_col(df, "logical_gap_value")
    forced = _np_col(df, "forced_gap_value")
    logical_reject = logical <= threshold
    forced_reject = forced <= threshold
    return {
        "both_rejected": np.logical_and(logical_reject, forced_reject),
        "overconfident": np.logical_and(logical_reject, ~forced_reject),
        "underconfident": np.logical_and(~logical_reject, forced_reject),
    }


def compute_gap_threshold_rejection_comparison(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    *,
    thresholds: Sequence[float] | None = None,
    forced_logical_error: bool | None = None,
) -> pl.DataFrame:
    """Compare logical/forced rejected sets at logical-gap value thresholds.

    For each threshold ``t``, shots with ``gap <= t`` are rejected independently
    by exact logical gap and forced gap. Fractions are normalized by the number
    of shots rejected by forced gap at that same threshold.
    """
    df = collect_ranking_shot_table(manager, config)
    if forced_logical_error is not None:
        df = df.filter(pl.col("forced_is_logical_error") == forced_logical_error)
    threshold_values = _logical_gap_thresholds(df, thresholds)
    n_total = df.height
    rows: list[dict[str, Any]] = []

    for t in threshold_values:
        masks = _threshold_rejection_masks(df, float(t))
        forced_reject_total = int(
            (masks["both_rejected"] | masks["underconfident"]).sum()
        )
        logical_reject_total = int(
            (masks["both_rejected"] | masks["overconfident"]).sum()
        )
        for set_name, mask in masks.items():
            n = int(mask.sum())
            rows.append(
                {
                    "threshold": float(t),
                    "set_name": set_name,
                    "set_label": _threshold_set_label(set_name),
                    "n": n,
                    "n_total": n_total,
                    "forced_reject_total": forced_reject_total,
                    "logical_reject_total": logical_reject_total,
                    "forced_logical_error_filter": forced_logical_error,
                    "fraction_of_forced_reject": (
                        n / forced_reject_total
                        if forced_reject_total > 0
                        else np.nan
                    ),
                    "fraction_of_all": n / n_total if n_total > 0 else np.nan,
                }
            )
    return pl.DataFrame(rows)


def compute_gap_threshold_rejection_case_fractions(
    manager: SimulationDataManager,
    config: RankingAnalysisConfig,
    *,
    thresholds: Sequence[float] | None = None,
) -> pl.DataFrame:
    """Compute stage/override category fractions for threshold rejection sets."""
    df = collect_ranking_shot_table(manager, config)
    threshold_values = _logical_gap_thresholds(df, thresholds)
    same = _np_col(df, "stage1_weight_same").astype(bool)
    override = _np_col(df, "forced_override").astype(bool)
    forced_errors = _np_col(df, "forced_is_logical_error").astype(bool)
    categories: dict[str, np.ndarray] = {
        "stage1_same_weight": same,
        "stage1_different_weight_override": np.logical_and(~same, override),
        "stage1_different_weight_no_override": np.logical_and(~same, ~override),
    }
    rows: list[dict[str, Any]] = []

    for t in threshold_values:
        set_masks = _threshold_rejection_masks(df, float(t))
        for set_name, set_mask in set_masks.items():
            n_set = int(set_mask.sum())
            for category, category_mask in categories.items():
                mask = np.logical_and(set_mask, category_mask)
                n = int(mask.sum())
                rows.append(
                    {
                        "threshold": float(t),
                        "set_name": set_name,
                        "set_label": _threshold_set_label(set_name),
                        "category": category,
                        "category_label": _threshold_category_label(category),
                        "n": n,
                        "n_set": n_set,
                        "fraction_within_set": n / n_set if n_set > 0 else np.nan,
                        "forced_logical_error_rate": (
                            float(forced_errors[mask].mean()) if n > 0 else np.nan
                        ),
                    }
                )
    return pl.DataFrame(rows)


class RankingDegradationAnalyzer:
    """Plotting helper for ranking/post-selection degradation analyses."""

    def plot_ranking_degradation(
        self,
        manager: SimulationDataManager,
        config: RankingAnalysisConfig,
        ax: plt.Axes,
        *,
        metrics: Sequence[RankingMetricName] = (
            "forced_only_fraction",
            "jaccard_distance",
            "symmetric_difference_fraction",
        ),
        exact_score: ScoreName = "exact_gap",
        comparison_score: ScoreName = "forced_gap",
    ) -> plt.Axes:
        df = compute_ranking_degradation(
            manager,
            config,
            exact_score=exact_score,
            comparison_score=comparison_score,
        )
        for metric in metrics:
            ax.plot(
                df["abort_rate"].to_numpy(),
                df[metric].to_numpy(),
                marker="o",
                label=_ranking_metric_label(metric),
            )
        ax.set_xlabel("Abort rate")
        ax.set_ylabel("Fraction / distance")
        ax.legend()
        return ax

    def plot_gap_threshold_rejection_fractions(
        self,
        manager: SimulationDataManager,
        config: RankingAnalysisConfig,
        ax: plt.Axes,
        *,
        thresholds: Sequence[float] | None = None,
        forced_logical_error: bool | None = None,
        sets: Sequence[ThresholdRejectionSet] = (
            "both_rejected",
            "overconfident",
            "underconfident",
        ),
    ) -> plt.Axes:
        df = compute_gap_threshold_rejection_comparison(
            manager,
            config,
            thresholds=thresholds,
            forced_logical_error=forced_logical_error,
        )
        for set_name in sets:
            sub = df.filter(pl.col("set_name") == set_name)
            ax.scatter(
                sub["threshold"].to_numpy(),
                sub["fraction_of_forced_reject"].to_numpy(),
                label=_threshold_set_label(set_name),
            )
            ax.plot(
                sub["threshold"].to_numpy(),
                sub["fraction_of_forced_reject"].to_numpy(),
                alpha=0.45,
            )
        ax.set_xlabel(r"Logical-gap threshold")
        ax.set_ylabel("Fraction / forced-gap rejected shots")
        if forced_logical_error is True:
            ax.set_title("Forced-gap logical-error shots")
        elif forced_logical_error is False:
            ax.set_title("Forced-gap non-logical-error shots")
        ax.legend()
        return ax

    def plot_gap_threshold_case_fractions(
        self,
        manager: SimulationDataManager,
        config: RankingAnalysisConfig,
        ax: plt.Axes,
        *,
        threshold: float,
    ) -> plt.Axes:
        df = compute_gap_threshold_rejection_case_fractions(
            manager,
            config,
            thresholds=[threshold],
        )
        set_names = ["both_rejected", "overconfident", "underconfident"]
        categories = [
            "stage1_same_weight",
            "stage1_different_weight_override",
            "stage1_different_weight_no_override",
        ]
        x = np.arange(len(set_names), dtype=float)
        width = min(0.8 / len(categories), 0.8)

        for i, category in enumerate(categories):
            sub = df.filter(pl.col("category") == category)
            value_map = dict(zip(sub["set_name"].to_list(), sub["fraction_within_set"].to_list()))
            ler_map = dict(
                zip(sub["set_name"].to_list(), sub["forced_logical_error_rate"].to_list())
            )
            heights = [value_map.get(name, np.nan) for name in set_names]
            lers = [ler_map.get(name, np.nan) for name in set_names]
            offset = (i - (len(categories) - 1) / 2.0) * width
            centers = x + offset
            ax.bar(
                centers,
                heights,
                width=width,
                label=_threshold_category_label(category),
                alpha=0.85,
            )
            for center, height, ler in zip(centers, heights, lers):
                if np.isfinite(height) and np.isfinite(ler):
                    ax.hlines(
                        height * ler,
                        center - width / 2.0,
                        center + width / 2.0,
                        colors="black",
                        linestyles=":",
                        linewidth=1.2,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([_threshold_set_label(name) for name in set_names], rotation=15)
        ax.set_xlabel("Rejection-set relation")
        ax.set_ylabel("Fraction within set")
        ax.set_title(rf"Logical-gap threshold $= {threshold:g}$")
        ax.plot(
            [],
            [],
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="forced-gap logical error rate within each bar",
        )
        ax.legend()
        return ax

    def plot_postselection_curves(
        self,
        manager: SimulationDataManager,
        config: RankingAnalysisConfig,
        ax: plt.Axes,
        *,
        error_source: Literal["score_default", "logical", "forced"] = "score_default",
        shade_ci: bool = True,
    ) -> plt.Axes:
        df = compute_postselection_curves(manager, config, error_source=error_source)
        for score in config.scores:
            sub = df.filter(pl.col("score") == score)
            x = sub["abort_rate"].to_numpy()
            y = sub["post_ler"].to_numpy()
            line = ax.plot(x, y, marker="o", label=_score_label(score))[0]
            if shade_ci:
                ax.fill_between(
                    x,
                    sub["ci_low"].to_numpy(),
                    sub["ci_high"].to_numpy(),
                    color=line.get_color(),
                    alpha=0.15,
                    linewidth=0,
                )
        ax.set_xlabel("Abort rate")
        ax.set_ylabel("Post-selected logical error rate")
        ax.set_yscale("log")
        ax.legend()
        return ax

    def plot_false_safe_case_fractions(
        self,
        manager: SimulationDataManager,
        config: RankingAnalysisConfig,
        ax: plt.Axes,
        *,
        abort_rates: Sequence[float],
        exact_score: ScoreName = "exact_gap",
        comparison_score: ScoreName = "forced_gap",
    ) -> plt.Axes:
        df = compute_false_safe_indicator_fractions(
            manager,
            config,
            abort_rates=abort_rates,
            exact_score=exact_score,
            comparison_score=comparison_score,
        )
        rates = df["abort_rate"].unique().sort().to_list()
        indicators = df["indicator"].unique(maintain_order=True).to_list()
        x = np.arange(len(rates), dtype=float)
        width = min(0.8 / max(len(indicators), 1), 0.8)

        for i, indicator in enumerate(indicators):
            sub = df.filter(pl.col("indicator") == indicator)
            values = dict(zip(sub["abort_rate"].to_list(), sub["fraction"].to_list()))
            heights = [values.get(r, np.nan) for r in rates]
            offset = (i - (len(indicators) - 1) / 2.0) * width
            ax.bar(x + offset, heights, width=width, label=indicator, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:g}" for r in rates])
        ax.set_xlabel("Abort rate")
        ax.set_ylabel("Fraction within forced-only accepted")
        ax.legend()
        return ax
