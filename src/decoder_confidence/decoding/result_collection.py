from __future__ import annotations

import contextlib
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


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX fallback
        fcntl = None

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _write_merged_decoder_stats(
    scan: pl.LazyFrame,
    output_dir: Path,
    batch_num: int,
    decoder_cols: list[str],
) -> None:
    if not decoder_cols:
        return

    target = output_dir / "decoder_stat.parquet"
    lock_path = output_dir / ".decoder_stat.lock"
    batch_tmp = output_dir / f".decoder_stat_batch={batch_num}.tmp.parquet"
    merge_tmp = output_dir / ".decoder_stat.tmp.parquet"

    exprs: list[pl.Expr | str] = [
        "shot_id",
        pl.lit(batch_num, dtype=pl.Int64).alias("batch"),
    ]
    exprs.extend(
        pl.col(col).alias(col.removeprefix(DECODER_STAT_PREFIX))
        for col in decoder_cols
    )
    scan.select(exprs).sort(["batch", "shot_id"]).sink_parquet(
        batch_tmp,
        compression="zstd",
    )

    with _file_lock(lock_path):
        new_lf = pl.scan_parquet(batch_tmp)
        if target.exists():
            old_lf = pl.scan_parquet(target).filter(pl.col("batch") != batch_num)
            merged = pl.concat([old_lf, new_lf], how="diagonal")
        else:
            merged = new_lf
        merged.sort(["batch", "shot_id"]).sink_parquet(merge_tmp, compression="zstd")
        merge_tmp.replace(target)

    with contextlib.suppress(OSError):
        batch_tmp.unlink()


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
    _write_merged_decoder_stats(
        scan,
        output_dir,
        batch_num,
        _prefixed_cols(schema, DECODER_STAT_PREFIX),
    )

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
