from __future__ import annotations

import contextlib
import re
from pathlib import Path

import polars as pl

from decoder_confidence.execution.worker import DECODER_STAT_PREFIX, DETAIL_STAT_PREFIX
from decoder_confidence.varint import write_obs_flip_idx_file

DETAIL_STAT_COLUMN_RENAMES: dict[str, str] = {
    "stage1_obs_flip": "baseline_logical_error",
    "stage1_weight": "baseline_correction_weight",
    "stage2_obs_flip": "forced_logical_error",
    "stage2_weight": "forced_correction_weight",
    "stage2_2ndbest_obs_flip": "forced_2nd_best_logical_error",
    "stage2_2nd_best_obs_flip": "forced_2nd_best_logical_error",
    "stage2_2ndbest_weight": "forced_2nd_best_correction_weight",
    "stage2_2nd_best_weight": "forced_2nd_best_correction_weight",
}

_LEGACY_DETAIL_BATCH_RE = re.compile(r"_batch=(\d+)\.parquet$")


def _prefixed_cols(schema: pl.Schema, prefix: str) -> list[str]:
    return sorted(name for name in schema if name.startswith(prefix))


def _write_detailed_stats(
    scan: pl.LazyFrame,
    output_dir: Path,
    batch_num: int,
    detail_cols: list[str],
) -> None:
    if not detail_cols:
        return

    used_names: set[str] = set()
    exprs: list[pl.Expr | str] = ["shot_id"]
    for col in detail_cols:
        stat_name = col.removeprefix(DETAIL_STAT_PREFIX)
        output_name = DETAIL_STAT_COLUMN_RENAMES.get(stat_name, stat_name)
        if output_name in used_names:
            continue
        used_names.add(output_name)
        exprs.append(pl.col(col).alias(output_name))

    scan.select(exprs).sink_parquet(
        output_dir / f"detailed_stats_batch={batch_num}.parquet",
        compression="zstd",
    )


def _write_decoder_stats(
    scan: pl.LazyFrame,
    output_dir: Path,
    batch_num: int,
    decoder_cols: list[str],
) -> None:
    if not decoder_cols:
        return

    exprs: list[pl.Expr | str] = [
        "shot_id",
    ]
    exprs.extend(
        pl.col(col).alias(col.removeprefix(DECODER_STAT_PREFIX))
        for col in decoder_cols
    )
    scan.select(exprs).sort("shot_id").sink_parquet(
        output_dir / f"decoder_stat_batch={batch_num}.parquet",
        compression="zstd",
    )


def _extract_legacy_batch(path: Path) -> int | None:
    match = _LEGACY_DETAIL_BATCH_RE.search(path.name)
    return int(match.group(1)) if match else None


def _legacy_detail_files_by_batch(output_dir: Path) -> dict[int, dict[str, Path]]:
    by_batch: dict[int, dict[str, Path]] = {}
    for legacy_name in DETAIL_STAT_COLUMN_RENAMES:
        candidates = sorted(output_dir.glob(f"metric={legacy_name}_batch=*.parquet"))
        candidates.extend(sorted(output_dir.glob(f"{legacy_name}_batch=*.parquet")))
        for path in candidates:
            batch_idx = _extract_legacy_batch(path)
            if batch_idx is None:
                continue
            by_batch.setdefault(batch_idx, {}).setdefault(legacy_name, path)
    return by_batch


def convert_legacy_detail_stats(output_dir: Path) -> None:
    """Write new-format detail files for any legacy per-field detail outputs.

    Legacy files are left untouched.  Existing new-format files are also left
    untouched so a just-finished batch cannot be overwritten by stale data.
    """
    for batch_idx, legacy_files in sorted(_legacy_detail_files_by_batch(output_dir).items()):
        target = output_dir / f"detailed_stats_batch={batch_idx}.parquet"
        if target.exists() or not legacy_files:
            continue

        joined: pl.LazyFrame | None = None
        used_names: set[str] = set()
        for legacy_name, path in sorted(legacy_files.items()):
            output_name = DETAIL_STAT_COLUMN_RENAMES.get(legacy_name, legacy_name)
            if output_name in used_names:
                continue
            schema = pl.scan_parquet(path).collect_schema().names()
            if legacy_name not in schema:
                continue
            lf = pl.scan_parquet(path).select(
                "shot_id",
                pl.col(legacy_name).alias(output_name),
            )
            joined = lf if joined is None else joined.join(lf, on="shot_id", how="inner")
            used_names.add(output_name)

        if joined is not None:
            joined.sort("shot_id").sink_parquet(target, compression="zstd")


def collect_results(
    chunk_dir: Path,
    output_dir: Path,
    batch_num: int,
    metric: str,
) -> list[str]:
    chunk_paths = sorted(chunk_dir.glob("chunk_*.parquet"))
    part_paths = sorted((chunk_dir / "parts").glob("part_*.parquet"))
    input_paths = chunk_paths + part_paths
    if not input_paths:
        raise FileNotFoundError(f"No chunk/part parquet files found in {chunk_dir}")

    scan = pl.scan_parquet([str(path) for path in input_paths])

    scan.select(["shot_id", "is_logical_error"]).sink_parquet(
        output_dir / f"logicalerror_batch={batch_num}.parquet",
        compression="zstd",
    )

    schema = scan.collect_schema()
    metric_cols = _prefixed_cols(schema, "metric_")
    if not metric_cols:
        raise ValueError("No metric columns found in chunk outputs")

    metric_names = sorted({name.removeprefix("metric_") for name in metric_cols})
    if metric not in metric_names:
        available = ", ".join(metric_names)
        raise ValueError(
            f"Configured metric '{metric}' missing in chunk outputs (found: {available})"
        )

    for metric_name in metric_names:
        metric_col = f"metric_{metric_name}"
        scan.select(["shot_id", pl.col(metric_col).alias(metric_name)]).sink_parquet(
            output_dir / f"metric={metric_name}_batch={batch_num}.parquet",
            compression="zstd",
        )

    _write_detailed_stats(
        scan,
        output_dir,
        batch_num,
        _prefixed_cols(schema, DETAIL_STAT_PREFIX),
    )
    _write_decoder_stats(
        scan,
        output_dir,
        batch_num,
        _prefixed_cols(schema, DECODER_STAT_PREFIX),
    )
    convert_legacy_detail_stats(output_dir)

    if "obs_flip_idx" in schema:
        obs_df = scan.select(["shot_id", "obs_flip_idx"]).sort("shot_id").collect()
        write_obs_flip_idx_file(
            output_dir / f"obs_flip_idx_batch={batch_num}.bin",
            obs_df["obs_flip_idx"].to_list(),
        )

    return metric_names


def cleanup_intermediate(chunk_dir: Path) -> None:
    for path in chunk_dir.glob("chunk_*.parquet"):
        with contextlib.suppress(OSError):
            path.unlink()

    part_dir = chunk_dir / "parts"
    if part_dir.exists():
        for path in part_dir.glob("part_*.parquet"):
            with contextlib.suppress(OSError):
                path.unlink()
        with contextlib.suppress(OSError):
            part_dir.rmdir()
