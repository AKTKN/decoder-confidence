from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import polars as pl

from analysis.src.data_manager import (
    SimulationDataManager,
    _extract_batch_idx,
    _infer_literal,
    _is_circuit_params_dir,
    _matches_all_filters,
    _parse_kv,
    _strip_quotes,
)

Stage1WeightCase = Literal["all", "same", "different", "diff"]
OverrideCase = Literal["all", "override", "no_override"]
OverrideThresholdGap = Literal["forced_gap_ml", "logical_gap", "anchored_gap"]
GapDifferenceStat = Literal[
    "ME",
    "MAE",
    "RMSE",
    "P(overestimate)",
    "P(underestimate)",
    "P(exact match)",
]

STAT_LABELS: dict[str, str] = {
    "ME": r"$\mathrm{E}[\Delta_{\rm fg}-\Delta_{\rm ex}]$",
    "MAE": r"$\mathrm{E}[|\Delta_{\rm fg}-\Delta_{\rm ex}|]$",
    "RMSE": r"$\sqrt{\mathrm{E}[(\Delta_{\rm fg}-\Delta_{\rm ex})^2]}$",
    "P(overestimate)": r"$P(\Delta_{\rm fg}>\Delta_{\rm ex})$",
    "P(underestimate)": r"$P(\Delta_{\rm fg}<\Delta_{\rm ex})$",
    "P(exact match)": r"$P(\Delta_{\rm fg}=\Delta_{\rm ex})$",
}


@dataclass
class Stage1GapDifferenceConfig:
    """Configuration for stage-1 forced-gap vs exact logical-gap analysis.

    ``logical_gap`` is read from detailed ``metric=logical_gap,get_detail_stat=True``
    directories, while ``forced_gap_ml`` is read from detailed
    ``metric=forced_gap_ml,...,get_detail_stat=True`` directories.  The two
    result tables are joined shot-by-shot within the same circuit parameters
    and batch.
    """

    logical_decoder_names: list[str] | None = None
    forced_decoder_names: list[str] | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    batch_indices: list[int] | None = None
    stage1_weight_case: Stage1WeightCase = "same"
    gap_round_digits: int | None = None
    compare_tolerance: float = 1e-9


def _decoder_allowed(decoder: str, names: list[str] | None) -> bool:
    return names is None or decoder in names


def _normalize_stage1_weight_case(case: str) -> Literal["all", "same", "different"]:
    if case == "diff":
        return "different"
    if case in {"all", "same", "different"}:
        return case  # type: ignore[return-value]
    raise ValueError("stage1_weight_case must be one of 'all', 'same', 'different', or 'diff'")


def _case_display_label(case: str) -> str:
    return "diff" if _normalize_stage1_weight_case(case) == "different" else case


def _has_truthy_param(params: dict[str, str], key: str) -> bool:
    return _strip_quotes(params.get(key, "")).lower() == "true"


def _scan_metric_file(metric_file: Path, batch_idx: int) -> pl.LazyFrame:
    metric_name = metric_file.name.removeprefix("metric=").split("_batch=", 1)[0]
    return pl.scan_parquet(metric_file).rename({metric_name: metric_name})


def _join_optional_metric(
    base: pl.LazyFrame,
    dm_dir: Path,
    metric_name: str,
    batch_idx: int,
    alias: str,
    *,
    required: bool,
) -> pl.LazyFrame:
    metric_file = dm_dir / f"metric={metric_name}_batch={batch_idx}.parquet"
    if not metric_file.exists():
        if required:
            raise FileNotFoundError(f"Required detailed metric file not found: {metric_file}")
        return base

    detail_lf = pl.scan_parquet(metric_file)
    if metric_name not in detail_lf.collect_schema().names():
        raise ValueError(f"Expected column {metric_name!r} in {metric_file}")
    detail_lf = detail_lf.select("shot_id", pl.col(metric_name).alias(alias))
    return base.join(detail_lf, on="shot_id", how="inner")


def _join_logical_error(
    base: pl.LazyFrame,
    dm_dir: Path,
    batch_idx: int,
    *,
    prefix: str,
) -> pl.LazyFrame:
    le_file = dm_dir / f"logicalerror_batch={batch_idx}.parquet"
    if not le_file.exists():
        raise FileNotFoundError(f"Required logical-error file not found: {le_file}")

    le_lf = pl.scan_parquet(le_file)
    if "is_logical_error" not in le_lf.collect_schema().names():
        raise ValueError(f"Expected column 'is_logical_error' in {le_file}")
    le_lf = le_lf.select(
        "shot_id",
        pl.col("is_logical_error").alias(f"{prefix}_is_logical_error"),
    )
    return base.join(le_lf, on="shot_id", how="inner")


def _load_metric_detail_lazy(
    dm_dir: Path,
    metric_name: str,
    batch_idx: int,
    all_params: dict[str, str],
    *,
    prefix: str,
) -> pl.LazyFrame:
    metric_file = dm_dir / f"metric={metric_name}_batch={batch_idx}.parquet"
    if not metric_file.exists():
        raise FileNotFoundError(f"Required metric file not found: {metric_file}")

    lf = pl.scan_parquet(metric_file).select(
        "shot_id",
        pl.col(metric_name).alias(metric_name),
    )
    lf = _join_logical_error(lf, dm_dir, batch_idx, prefix=prefix)
    lf = _join_optional_metric(
        lf,
        dm_dir,
        "stage1_weight",
        batch_idx,
        f"{prefix}_stage1_weight",
        required=True,
    )
    lf = _join_optional_metric(
        lf,
        dm_dir,
        "stage1_obs_flip",
        batch_idx,
        f"{prefix}_stage1_obs_flip",
        required=False,
    )
    lf = _join_optional_metric(
        lf,
        dm_dir,
        "stage2_weight",
        batch_idx,
        f"{prefix}_stage2_weight",
        required=False,
    )
    lf = _join_optional_metric(
        lf,
        dm_dir,
        "stage2_obs_flip",
        batch_idx,
        f"{prefix}_stage2_obs_flip",
        required=False,
    )

    if metric_name == "forced_gap_ml":
        lf = _join_optional_metric(
            lf,
            dm_dir,
            "stage2_2ndbest_weight",
            batch_idx,
            f"{prefix}_stage2_2ndbest_weight",
            required=False,
        )
        lf = _join_optional_metric(
            lf,
            dm_dir,
            "stage2_2ndbest_obs_flip",
            batch_idx,
            f"{prefix}_stage2_2ndbest_obs_flip",
            required=False,
        )

    decoder = _strip_quotes(all_params.get("decoder", ""))
    literal_exprs = [
        _infer_literal(val).alias(key) for key, val in all_params.items()
    ]
    literal_exprs.extend(
        [
            pl.lit(batch_idx, dtype=pl.Int64).alias("batch"),
            pl.lit(decoder).alias(f"{prefix}_decoder"),
        ]
    )
    return lf.with_columns(literal_exprs)


def _find_detail_dirs(
    circuit_dir: Path,
    metric_name: str,
    decoder_names: list[str] | None,
) -> list[tuple[Path, dict[str, str]]]:
    decoding_dir = circuit_dir / "decoding_result"
    if not decoding_dir.is_dir():
        return []

    result: list[tuple[Path, dict[str, str]]] = []
    for entry in sorted(decoding_dir.iterdir()):
        if not entry.is_dir():
            continue
        params = _parse_kv(entry.name)
        if _strip_quotes(params.get("metric", "")) != metric_name:
            continue
        if not _has_truthy_param(params, "get_detail_stat"):
            continue

        decoder = _strip_quotes(params.get("decoder", ""))
        if not _decoder_allowed(decoder, decoder_names):
            continue
        result.append((entry, params))
    return result


def load_stage1_gap_difference_lazy(
    result_dir_root: Path,
    config: Stage1GapDifferenceConfig,
) -> pl.LazyFrame:
    """Load exact/forced detailed gap data joined shot-by-shot."""
    frames: list[pl.LazyFrame] = []

    for circuit_entry in sorted(result_dir_root.iterdir()):
        if not circuit_entry.is_dir() or not _is_circuit_params_dir(circuit_entry.name):
            continue
        circuit_params = _parse_kv(circuit_entry.name)
        if not _matches_all_filters(circuit_params, config.filters):
            continue

        logical_dirs = _find_detail_dirs(
            circuit_entry, "logical_gap", config.logical_decoder_names
        )
        forced_dirs = _find_detail_dirs(
            circuit_entry, "forced_gap_ml", config.forced_decoder_names
        )

        for logical_dir, logical_params in logical_dirs:
            for forced_dir, forced_params in forced_dirs:
                for logical_file in sorted(logical_dir.glob("metric=logical_gap_batch=*.parquet")):
                    batch_idx = _extract_batch_idx(logical_file.name)
                    if batch_idx is None:
                        continue
                    if config.batch_indices is not None and batch_idx not in config.batch_indices:
                        continue
                    forced_file = forced_dir / f"metric=forced_gap_ml_batch={batch_idx}.parquet"
                    if not forced_file.exists():
                        continue

                    logical_lf = _load_metric_detail_lazy(
                        logical_dir,
                        "logical_gap",
                        batch_idx,
                        {**circuit_params, **logical_params},
                        prefix="logical",
                    )
                    forced_lf = _load_metric_detail_lazy(
                        forced_dir,
                        "forced_gap_ml",
                        batch_idx,
                        {**circuit_params, **forced_params},
                        prefix="forced",
                    )

                    join_keys = ["shot_id", "batch"]
                    joined = logical_lf.join(
                        forced_lf.select(
                            [
                                "shot_id",
                                "batch",
                                "forced_gap_ml",
                                "forced_is_logical_error",
                                "forced_stage1_weight",
                                "forced_stage1_obs_flip",
                                "forced_stage2_weight",
                                "forced_stage2_obs_flip",
                                "forced_stage2_2ndbest_weight",
                                "forced_stage2_2ndbest_obs_flip",
                                "forced_decoder",
                            ]
                        ),
                        on=join_keys,
                        how="inner",
                    )
                    frames.append(_with_derived_columns(joined, config))

    if not frames:
        raise FileNotFoundError(
            "No joined logical_gap / forced_gap_ml detail data found. "
            "Check filters, decoder names, batch_indices, and get_detail_stat=True outputs."
        )
    return pl.concat(frames, how="diagonal")


def _with_derived_columns(
    lf: pl.LazyFrame,
    config: Stage1GapDifferenceConfig,
) -> pl.LazyFrame:
    tol = float(config.compare_tolerance)
    exact = (pl.col("forced_gap_ml") - pl.col("logical_gap")).abs() <= tol
    same_stage1 = (
        pl.col("forced_stage1_weight") - pl.col("logical_stage1_weight")
    ).abs() <= tol
    logical_signed = (
        pl.when(pl.col("logical_is_logical_error"))
        .then(-pl.col("logical_gap"))
        .otherwise(pl.col("logical_gap"))
    )
    forced_signed = (
        pl.when(pl.col("forced_is_logical_error"))
        .then(-pl.col("forced_gap_ml"))
        .otherwise(pl.col("forced_gap_ml"))
    )

    exprs = [
        (pl.col("forced_gap_ml") - pl.col("logical_gap")).alias("gap_diff"),
        same_stage1.alias("stage1_weight_same"),
        logical_signed.alias("logical_gap_signed"),
        forced_signed.alias("forced_gap_signed"),
        pl.when(exact)
        .then(pl.lit("exact"))
        .when(pl.col("forced_gap_ml") > pl.col("logical_gap") + tol)
        .then(pl.lit("overestimate"))
        .otherwise(pl.lit("underestimate"))
        .alias("gap_relation"),
    ]
    if config.gap_round_digits is None:
        exprs.append(pl.col("logical_gap").alias("logical_gap_value"))
        exprs.append(pl.col("forced_gap_ml").alias("forced_gap_value"))
    else:
        exprs.extend(
            [
                pl.col("logical_gap")
                .round(config.gap_round_digits)
                .alias("logical_gap_value"),
                pl.col("forced_gap_ml")
                .round(config.gap_round_digits)
                .alias("forced_gap_value"),
            ]
        )

    result = lf.with_columns(exprs)
    case = _normalize_stage1_weight_case(config.stage1_weight_case)
    if case == "same":
        return result.filter(pl.col("stage1_weight_same"))
    if case == "different":
        return result.filter(~pl.col("stage1_weight_same"))
    if case == "all":
        return result


def _stats_exprs() -> list[pl.Expr]:
    diff = pl.col("gap_diff")
    relation = pl.col("gap_relation")
    return [
        pl.len().alias("n"),
        diff.mean().alias("ME"),
        diff.abs().mean().alias("MAE"),
        (diff.pow(2).mean().sqrt()).alias("RMSE"),
        (relation.eq("overestimate").cast(pl.Float64).mean()).alias("P(overestimate)"),
        (relation.eq("underestimate").cast(pl.Float64).mean()).alias("P(underestimate)"),
        (relation.eq("exact").cast(pl.Float64).mean()).alias("P(exact match)"),
    ]


def compute_stage1_gap_difference_table(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
) -> pl.DataFrame:
    """Return aggregate Δ_fg − Δ_ex statistics for the configured subset."""
    lf = load_stage1_gap_difference_lazy(manager.result_dir_root, config)
    group_keys = ["logical_decoder", "forced_decoder"]
    return (
        lf.group_by(group_keys)
        .agg(_stats_exprs())
        .with_columns(pl.lit(config.stage1_weight_case).alias("stage1_weight_case"))
        .select(["stage1_weight_case", *group_keys, *_stat_columns()])
        .sort(group_keys)
        .collect()
    )


def compute_stage1_gap_difference_by_logical_gap(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
) -> pl.DataFrame:
    """Return Δ_fg − Δ_ex statistics conditioned on each logical_gap value."""
    lf = load_stage1_gap_difference_lazy(manager.result_dir_root, config)
    group_keys = ["logical_decoder", "forced_decoder", "logical_gap_value"]
    return (
        lf.group_by(group_keys)
        .agg(_stats_exprs())
        .with_columns(pl.lit(config.stage1_weight_case).alias("stage1_weight_case"))
        .select(["stage1_weight_case", *group_keys, *_stat_columns()])
        .sort(group_keys)
        .collect()
    )


def compute_gap_difference_by_override_case(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
    override_cases: Sequence[OverrideCase] = ("all", "override", "no_override"),
) -> pl.DataFrame:
    """Return Δ_fg − Δ_ex statistics split by forced-side override case.

    Override is defined on the forced-gap detailed outputs as
    ``forced_stage2_weight < forced_stage1_weight``.
    """
    lf = load_stage1_gap_difference_lazy(manager.result_dir_root, config).with_columns(
        (pl.col("forced_stage2_weight") < pl.col("forced_stage1_weight")).alias(
            "forced_override"
        )
    )
    group_keys = ["logical_decoder", "forced_decoder"]
    frames: list[pl.LazyFrame] = []

    for case in override_cases:
        if case == "all":
            case_lf = lf
        elif case == "override":
            case_lf = lf.filter(pl.col("forced_override"))
        elif case == "no_override":
            case_lf = lf.filter(~pl.col("forced_override"))
        else:
            raise ValueError(
                "override_cases must contain only 'all', 'override', or 'no_override'"
            )

        frames.append(
            case_lf.group_by(group_keys)
            .agg(_stats_exprs())
            .with_columns(
                [
                    pl.lit(case).alias("override_case"),
                    pl.lit(config.stage1_weight_case).alias("stage1_weight_case"),
                ]
            )
            .select(
                [
                    "override_case",
                    "stage1_weight_case",
                    *group_keys,
                    *_stat_columns(),
                ]
            )
        )

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal").sort(["override_case", *group_keys]).collect()


def compute_forced_obs_flip_rates_by_override_case(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
    override_cases: Sequence[OverrideCase] = ("all", "override", "no_override"),
) -> pl.DataFrame:
    """Return forced_gap_ml stage-1/stage-2 obs-flip rates by override case."""
    lf = load_stage1_gap_difference_lazy(manager.result_dir_root, config).with_columns(
        (pl.col("forced_stage2_weight") < pl.col("forced_stage1_weight")).alias(
            "forced_override"
        )
    )
    group_keys = ["logical_decoder", "forced_decoder"]
    frames: list[pl.LazyFrame] = []

    for case in override_cases:
        if case == "all":
            case_lf = lf
        elif case == "override":
            case_lf = lf.filter(pl.col("forced_override"))
        elif case == "no_override":
            case_lf = lf.filter(~pl.col("forced_override"))
        else:
            raise ValueError(
                "override_cases must contain only 'all', 'override', or 'no_override'"
            )

        frames.append(
            case_lf.group_by(group_keys)
            .agg(
                [
                    pl.len().alias("n"),
                    pl.col("forced_stage1_obs_flip")
                    .cast(pl.Float64)
                    .mean()
                    .alias("stage1_obs_flip_rate"),
                    pl.col("forced_stage2_obs_flip")
                    .cast(pl.Float64)
                    .mean()
                    .alias("stage2_min_obs_flip_rate"),
                ]
            )
            .with_columns(
                [
                    pl.lit(case).alias("override_case"),
                    pl.lit(config.stage1_weight_case).alias("stage1_weight_case"),
                ]
            )
            .select(
                [
                    "override_case",
                    "stage1_weight_case",
                    *group_keys,
                    "n",
                    "stage1_obs_flip_rate",
                    "stage2_min_obs_flip_rate",
                ]
            )
        )

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal").sort(["override_case", *group_keys]).collect()


def compute_override_threshold_logical_error_curve(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
    *,
    gap_name: OverrideThresholdGap = "forced_gap_ml",
    thresholds: Sequence[float] | None = None,
) -> pl.DataFrame:
    """Return conditional logical-error rates inside override shots below thresholds.

    The data is restricted to forced-side override shots, defined as
    ``forced_stage2_weight < forced_stage1_weight``. For each threshold ``t``,
    the curve point is computed on override shots satisfying ``gap <= t``.

    The plotted logical-error rate uses forced-gap ML's default logical-error
    column, ``forced_is_logical_error``. The reference line is
    ``P(forced_stage1_obs_flip | override)``.
    """
    lf = load_stage1_gap_difference_lazy(manager.result_dir_root, config).with_columns(
        [
            (pl.col("forced_stage2_weight") < pl.col("forced_stage1_weight")).alias(
                "forced_override"
            ),
            (pl.col("forced_stage2_weight") - pl.col("forced_stage1_weight")).alias(
                "anchored_gap"
            ),
        ]
    )
    if gap_name not in {"forced_gap_ml", "logical_gap", "anchored_gap"}:
        raise ValueError("gap_name must be 'forced_gap_ml', 'logical_gap', or 'anchored_gap'")

    if gap_name == "forced_gap_ml":
        gap_expr = pl.col("forced_gap_value")
    elif gap_name == "logical_gap":
        gap_expr = pl.col("logical_gap_value")
    else:
        anchored = pl.col("anchored_gap")
        gap_expr = (
            anchored
            if config.gap_round_digits is None
            else anchored.round(config.gap_round_digits)
        )

    override_df = lf.filter(pl.col("forced_override")).select(
        [
            gap_expr.alias("gap_value"),
            "forced_is_logical_error",
            "forced_stage1_obs_flip",
        ]
    ).collect()
    if override_df.height == 0:
        return pl.DataFrame()

    if thresholds is None:
        threshold_values = override_df["gap_value"].unique().sort().to_list()
    else:
        threshold_values = [float(v) for v in thresholds]

    baseline = float(
        override_df["forced_stage1_obs_flip"].cast(pl.Float64).mean()
    )
    rows: list[dict[str, Any]] = []
    gap_values = override_df["gap_value"].to_numpy()
    is_error = override_df["forced_is_logical_error"].to_numpy().astype(bool)
    n_override = override_df.height

    for threshold in threshold_values:
        mask = gap_values <= threshold
        n = int(mask.sum())
        n_err = int(is_error[mask].sum()) if n > 0 else 0
        rows.append(
            {
                "gap_name": gap_name,
                "threshold": float(threshold),
                "n": n,
                "n_override": n_override,
                "fraction_within_override": n / n_override,
                "n_logical_error": n_err,
                "conditional_logical_error_rate": n_err / n if n > 0 else np.nan,
                "stage1_obs_flip_rate_given_override": baseline,
            }
        )
    return pl.DataFrame(rows)


def compute_stage1_logical_gap_distribution(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
    stage1_weight_cases: Sequence[Stage1WeightCase] = ("all",),
) -> pl.DataFrame:
    """Return the logical_gap distribution for one or more stage-1 weight cases."""
    frames: list[pl.DataFrame] = []
    for case in stage1_weight_cases:
        case_config = replace(config, stage1_weight_case=case)
        display_case = _case_display_label(case)
        lf = load_stage1_gap_difference_lazy(manager.result_dir_root, case_config)
        group_keys = ["logical_decoder", "forced_decoder", "logical_gap_value"]
        counts = (
            lf.group_by(group_keys)
            .agg(pl.len().alias("n"))
            .with_columns(pl.lit(display_case).alias("stage1_weight_case"))
            .with_columns(
                (
                    pl.col("n")
                    / pl.col("n").sum().over(
                        ["stage1_weight_case", "logical_decoder", "forced_decoder"]
                    )
                ).alias("fraction")
            )
            .select(["stage1_weight_case", *group_keys, "n", "fraction"])
            .sort(["stage1_weight_case", *group_keys])
            .collect()
        )
        frames.append(counts)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def compute_stage1_forced_gap_conditional_distribution(
    manager: SimulationDataManager,
    config: Stage1GapDifferenceConfig,
    *,
    signed_by_logical_error: bool = False,
) -> pl.DataFrame:
    """Return P(Δ_fg | Δ_ex) for the configured stage-1 weight case."""
    lf = load_stage1_gap_difference_lazy(manager.result_dir_root, config)
    logical_value_col = "logical_gap_signed" if signed_by_logical_error else "logical_gap"
    forced_value_col = "forced_gap_signed" if signed_by_logical_error else "forced_gap_ml"
    if config.gap_round_digits is None:
        lf = lf.with_columns(
            [
                pl.col(logical_value_col).alias("logical_gap_value"),
                pl.col(forced_value_col).alias("forced_gap_value"),
            ]
        )
    else:
        lf = lf.with_columns(
            [
                pl.col(logical_value_col)
                .round(config.gap_round_digits)
                .alias("logical_gap_value"),
                pl.col(forced_value_col)
                .round(config.gap_round_digits)
                .alias("forced_gap_value"),
            ]
        )
    group_keys = [
        "logical_decoder",
        "forced_decoder",
        "logical_gap_value",
        "forced_gap_value",
    ]
    return (
        lf.group_by(group_keys)
        .agg(pl.len().alias("n"))
        .with_columns(
            pl.lit(_case_display_label(config.stage1_weight_case)).alias(
                "stage1_weight_case"
            ),
            pl.lit(signed_by_logical_error).alias("signed_by_logical_error"),
        )
        .with_columns(
            (
                pl.col("n")
                / pl.col("n").sum().over(
                    ["logical_decoder", "forced_decoder", "logical_gap_value"]
                )
            ).alias("conditional_probability")
        )
        .select(
            [
                "stage1_weight_case",
                "signed_by_logical_error",
                *group_keys,
                "n",
                "conditional_probability",
            ]
        )
        .sort(["logical_decoder", "forced_decoder", "logical_gap_value", "forced_gap_value"])
        .collect()
    )


def _value_edges(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot build heatmap edges from an empty value list.")
    if arr.size == 1:
        return np.asarray([arr[0] - 0.5, arr[0] + 0.5], dtype=float)

    mids = (arr[:-1] + arr[1:]) / 2.0
    first = arr[0] - (mids[0] - arr[0])
    last = arr[-1] + (arr[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _stat_columns() -> list[str]:
    return [
        "n",
        "ME",
        "MAE",
        "RMSE",
        "P(overestimate)",
        "P(underestimate)",
        "P(exact match)",
    ]


class Stage1GapDifferenceAnalyzer:
    """Analyzer and plotting helper for stage-1 gap-difference statistics."""

    def compute_table(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
    ) -> pl.DataFrame:
        return compute_stage1_gap_difference_table(manager, config)

    def compute_by_logical_gap(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
    ) -> pl.DataFrame:
        return compute_stage1_gap_difference_by_logical_gap(manager, config)

    def compute_by_override_case(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
        override_cases: Sequence[OverrideCase] = ("all", "override", "no_override"),
    ) -> pl.DataFrame:
        return compute_gap_difference_by_override_case(
            manager,
            config,
            override_cases=override_cases,
        )

    def compute_forced_obs_flip_rates_by_override_case(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
        override_cases: Sequence[OverrideCase] = ("all", "override", "no_override"),
    ) -> pl.DataFrame:
        return compute_forced_obs_flip_rates_by_override_case(
            manager,
            config,
            override_cases=override_cases,
        )

    def compute_override_threshold_logical_error_curve(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
        *,
        gap_name: OverrideThresholdGap = "forced_gap_ml",
        thresholds: Sequence[float] | None = None,
    ) -> pl.DataFrame:
        return compute_override_threshold_logical_error_curve(
            manager,
            config,
            gap_name=gap_name,
            thresholds=thresholds,
        )

    def plot_override_threshold_logical_error_curve(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
        ax: plt.Axes,
        *,
        gap_name: OverrideThresholdGap = "forced_gap_ml",
        thresholds: Sequence[float] | None = None,
        plot_kw: dict[str, Any] | None = None,
    ) -> plt.Axes:
        df = compute_override_threshold_logical_error_curve(
            manager,
            config,
            gap_name=gap_name,
            thresholds=thresholds,
        )
        if df.height == 0:
            raise ValueError("No override shots found for the given configuration.")

        plot_kw = {"marker": "o", **(plot_kw or {})}
        ax.plot(
            df["threshold"].to_numpy(),
            df["conditional_logical_error_rate"].to_numpy(),
            label="conditional forced logical error rate",
            **plot_kw,
        )
        baseline = float(df["stage1_obs_flip_rate_given_override"][0])
        ax.axhline(
            baseline,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=r"$P(\mathrm{stage1\ obs\ flip}\mid\mathrm{override})$",
        )
        gap_symbols = {
            "forced_gap_ml": r"\Delta_{\mathrm{fg}}",
            "logical_gap": r"\Delta_{\mathrm{ex}}",
            "anchored_gap": r"\Delta_{\mathrm{anchor}}",
        }
        gap_symbol = gap_symbols.get(gap_name, "g")
        ax.set_xlabel(rf"{gap_name} threshold $g_{{\mathrm{{th}}}}$")
        ax.set_ylabel(
            rf"$P(\mathrm{{logical\ error}}\mid \mathrm{{override}},\,"
            rf"{gap_symbol} \le g_{{\mathrm{{th}}}})$"
        )
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_anchor = x_min + 0.98 * (x_max - x_min)
        y_anchor = min(
            y_max,
            max(y_min, baseline + 0.04 * (y_max - y_min)),
        )
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(x_anchor, y_anchor),
            bbox_transform=ax.transData,
        )
        return ax

    def plot_stat_by_logical_gap(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
        ax: plt.Axes,
        *,
        stat_name: GapDifferenceStat = "MAE",
        bar_kw: dict[str, Any] | None = None,
    ) -> plt.Axes:
        if stat_name not in STAT_LABELS:
            valid = ", ".join(STAT_LABELS)
            raise ValueError(f"Unknown stat_name {stat_name!r}; expected one of {valid}")

        df = compute_stage1_gap_difference_by_logical_gap(manager, config)
        if df.height == 0:
            raise ValueError("No rows remain after applying stage1_weight_case filter.")

        bar_kw = {"alpha": 0.85, **(bar_kw or {})}
        pairs = (
            df.select("logical_decoder", "forced_decoder")
            .unique(maintain_order=True)
            .iter_rows()
        )
        pair_list = list(pairs)
        x_values = df["logical_gap_value"].unique().sort().to_list()
        x = np.arange(len(x_values), dtype=float)
        width = min(0.8 / max(len(pair_list), 1), 0.8)

        for i, (logical_decoder, forced_decoder) in enumerate(pair_list):
            sub = df.filter(
                (pl.col("logical_decoder") == logical_decoder)
                & (pl.col("forced_decoder") == forced_decoder)
            )
            value_map = dict(zip(sub["logical_gap_value"].to_list(), sub[stat_name].to_list()))
            heights = [value_map.get(v, np.nan) for v in x_values]
            offset = (i - (len(pair_list) - 1) / 2.0) * width
            label = f"{forced_decoder} vs {logical_decoder}"
            ax.bar(x + offset, heights, width=width, label=label, **bar_kw)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" if isinstance(v, float) else str(v) for v in x_values])
        ax.set_xlabel(r"$\Delta_{\rm ex}$")
        ax.set_ylabel(STAT_LABELS[stat_name])
        if len(pair_list) > 1:
            ax.legend()
        return ax

    def compute_logical_gap_distribution(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
        stage1_weight_cases: Sequence[Stage1WeightCase] = ("all",),
    ) -> pl.DataFrame:
        return compute_stage1_logical_gap_distribution(
            manager,
            config,
            stage1_weight_cases=stage1_weight_cases,
        )

    def compute_forced_gap_conditional_distribution(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
        *,
        signed_by_logical_error: bool = False,
    ) -> pl.DataFrame:
        return compute_stage1_forced_gap_conditional_distribution(
            manager,
            config,
            signed_by_logical_error=signed_by_logical_error,
        )

    def plot_logical_gap_distribution(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
        ax: plt.Axes,
        *,
        stage1_weight_cases: Sequence[Stage1WeightCase] = ("all",),
        normalize: bool = False,
        bar_kw: dict[str, Any] | None = None,
    ) -> plt.Axes:
        df = compute_stage1_logical_gap_distribution(
            manager,
            config,
            stage1_weight_cases=stage1_weight_cases,
        )
        if df.height == 0:
            raise ValueError("No rows remain after applying stage1_weight_cases.")

        value_col = "fraction" if normalize else "n"
        case_labels = [_case_display_label(case) for case in stage1_weight_cases]
        x_values = df["logical_gap_value"].unique().sort().to_list()
        x = np.arange(len(x_values), dtype=float)
        alpha = 0.45 if len(case_labels) > 1 else 0.85
        default_bar_kw: dict[str, Any] = {
            "width": 0.82,
            "alpha": alpha,
            "edgecolor": "black",
            "linewidth": 0.8,
        }
        default_bar_kw.update(bar_kw or {})

        pairs = (
            df.select("logical_decoder", "forced_decoder")
            .unique(maintain_order=True)
            .iter_rows()
        )
        pair_list = list(pairs)
        if len(pair_list) > 1:
            raise ValueError(
                "plot_logical_gap_distribution currently expects one logical/forced "
                "decoder pair. Restrict decoder names in the config."
            )

        for case in case_labels:
            sub = df.filter(pl.col("stage1_weight_case") == case)
            value_map = dict(zip(sub["logical_gap_value"].to_list(), sub[value_col].to_list()))
            heights = [value_map.get(v, 0.0) for v in x_values]
            ax.bar(x, heights, label=case, **default_bar_kw)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" if isinstance(v, float) else str(v) for v in x_values])
        ax.set_xlabel(r"$\Delta_{\rm ex}$")
        ax.set_ylabel("Fraction" if normalize else "Count")
        if len(case_labels) > 1:
            ax.legend(title="Stage-1 weight")
        return ax

    def plot_forced_gap_conditional_heatmap(
        self,
        manager: SimulationDataManager,
        config: Stage1GapDifferenceConfig,
        ax: plt.Axes,
        *,
        cmap: str = "viridis",
        add_colorbar: bool = True,
        colorbar_label: str = r"$P(\Delta_{\rm fg}\mid\Delta_{\rm ex})$",
        signed_by_logical_error: bool = False,
    ) -> plt.Axes:
        df = compute_stage1_forced_gap_conditional_distribution(
            manager,
            config,
            signed_by_logical_error=signed_by_logical_error,
        )
        if df.height == 0:
            raise ValueError("No rows remain after applying stage1_weight_case filter.")

        pairs = (
            df.select("logical_decoder", "forced_decoder")
            .unique(maintain_order=True)
            .iter_rows()
        )
        pair_list = list(pairs)
        if len(pair_list) > 1:
            raise ValueError(
                "plot_forced_gap_conditional_heatmap currently expects one "
                "logical/forced decoder pair. Restrict decoder names in the config."
            )

        x_values = df["logical_gap_value"].unique().sort().to_list()
        y_values = df["forced_gap_value"].unique().sort().to_list()
        x_index = {v: i for i, v in enumerate(x_values)}
        y_index = {v: i for i, v in enumerate(y_values)}

        matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
        for logical_gap, forced_gap, prob in df.select(
            "logical_gap_value", "forced_gap_value", "conditional_probability"
        ).iter_rows():
            matrix[y_index[forced_gap], x_index[logical_gap]] = prob

        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        if positive.size == 0:
            raise ValueError("No positive conditional probabilities to plot.")
        vmin = float(positive.min())
        vmax = float(positive.max())
        if vmin == vmax:
            vmin = max(vmin / 10.0, np.nextafter(0.0, 1.0))
            vmax = vmax * 10.0

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("white")
        mesh = ax.pcolormesh(
            _value_edges(x_values),
            _value_edges(y_values),
            np.ma.masked_invalid(matrix),
            cmap=cmap_obj,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="auto",
        )

        lo = min(float(min(x_values)), float(min(y_values)))
        hi = max(float(max(x_values)), float(max(y_values)))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=1.2)
        ax.set_xlim(float(_value_edges(x_values)[0]), float(_value_edges(x_values)[-1]))
        ax.set_ylim(float(_value_edges(y_values)[0]), float(_value_edges(y_values)[-1]))
        if signed_by_logical_error:
            ax.set_xlabel(r"signed $\Delta_{\rm ex}$")
            ax.set_ylabel(r"signed $\Delta_{\rm fg}$")
        else:
            ax.set_xlabel(r"$\Delta_{\rm ex}$")
            ax.set_ylabel(r"$\Delta_{\rm fg}$")

        if add_colorbar:
            colorbar = ax.figure.colorbar(mesh, ax=ax)
            colorbar.set_label(colorbar_label)
        return ax
