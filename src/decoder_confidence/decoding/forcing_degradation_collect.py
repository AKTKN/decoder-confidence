from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from decoder_confidence.decoding.__main__ import (
    _find_circuit_dir,
    _format_metric_options,
    parse_args,
)
from decoder_confidence.decoding._forcing_degradation_test import (
    FORCING_DEGRADATION_METRIC,
)
from decoder_confidence.decoding.decoder_factory import load_decoder_factory
from decoder_confidence.decoding.incomplete import (
    INCOMPLETE_SHOTS_FILENAME,
    write_incomplete_shots,
)
from decoder_confidence.decoding.metadata import (
    build_decoding_metadata,
    metadata_file_path,
    write_metadata,
)
from decoder_confidence.decoding.result_collection import cleanup_intermediate
from decoder_confidence.execution.manager import ExecutionConfig, run_manager
from decoder_confidence.execution.models import IncompleteTasksError


def _collect_forcing_degradation_results(
    chunk_dir: Path,
    output_dir: Path,
    batch_num: int,
) -> list[str]:
    """Finalize forcing-degradation chunk outputs into batch Parquet files.

    Parameters
    ----------
    chunk_dir:
        Directory containing worker-produced ``chunk_*.parquet`` files and
        optional merged ``parts/part_*.parquet`` files.
    output_dir:
        Final decoder/metric result directory.
    batch_num:
        One-based sampled-data batch index.

    Returns
    -------
    list[str]
        Names of shot-level result columns recorded in
        ``metric=forcing_degradation_test_batch=<batch_num>.parquet`` excluding
        ``shot_id``.

    Raises
    ------
    FileNotFoundError
        If no worker chunks are available.
    ValueError
        If required forcing-degradation metric columns are missing.
    """
    chunk_paths = sorted(chunk_dir.glob("chunk_*.parquet"))
    part_paths = sorted((chunk_dir / "parts").glob("part_*.parquet"))
    input_paths = chunk_paths + part_paths
    if not input_paths:
        raise FileNotFoundError(f"No chunk/part parquet files found in {chunk_dir}")

    scan = pl.scan_parquet([str(path) for path in input_paths])
    schema = scan.collect_schema()
    metric_cols = sorted(name for name in schema if name.startswith("metric_"))

    required = {"metric_baseline_weight", "metric_forced_weight"}
    missing = sorted(required - set(metric_cols))
    if missing:
        raise ValueError(
            "forcing_degradation_test chunk outputs are missing required columns: "
            + ", ".join(missing)
        )

    stage_metric_cols = [
        name
        for name in metric_cols
        if name.startswith("metric_baseline_")
        and name not in {"metric_baseline_weight"}
    ]
    if len(stage_metric_cols) != 1:
        raise ValueError(
            "Expected exactly one baseline decoder-specific metric column, found: "
            + ", ".join(stage_metric_cols)
        )
    stage_suffix = stage_metric_cols[0].removeprefix("metric_baseline_")
    forced_stage_col = f"metric_forced_{stage_suffix}"
    if forced_stage_col not in metric_cols:
        raise ValueError(
            f"Missing forced decoder-specific metric column {forced_stage_col!r}"
        )

    result_columns = [
        "logicalerror",
        "baseline_weight",
        "forced_weight",
        f"baseline_{stage_suffix}",
        f"forced_{stage_suffix}",
    ]
    result_exprs = [
        pl.col("shot_id"),
        pl.col("is_logical_error").cast(pl.Boolean).alias("logicalerror"),
        pl.col("metric_baseline_weight").cast(pl.Float64).alias("baseline_weight"),
        pl.col("metric_forced_weight").cast(pl.Float64).alias("forced_weight"),
    ]
    if stage_suffix == "iteration":
        result_exprs.extend(
            [
                pl.col(stage_metric_cols[0]).cast(pl.Int64).alias("baseline_iteration"),
                pl.col(forced_stage_col).cast(pl.Int64).alias("forced_iteration"),
            ]
        )
    else:
        result_exprs.extend(
            [
                pl.col(stage_metric_cols[0]).cast(pl.Float64).alias(
                    f"baseline_{stage_suffix}"
                ),
                pl.col(forced_stage_col).cast(pl.Float64).alias(
                    f"forced_{stage_suffix}"
                ),
            ]
        )

    logicalerror_path = output_dir / f"logicalerror_batch={batch_num}.parquet"
    scan.select(
        ["shot_id", pl.col("is_logical_error").cast(pl.Boolean)]
    ).sink_parquet(logicalerror_path, compression="zstd")

    metric_path = output_dir / f"metric={FORCING_DEGRADATION_METRIC}_batch={batch_num}.parquet"
    scan.select(result_exprs).sink_parquet(metric_path, compression="zstd")
    return result_columns


def main(argv: list[str] | None = None) -> int:
    """Run the forcing-degradation collection workflow for one sampled batch."""
    args = parse_args(argv)

    if args.batch_num < 1:
        raise ValueError("batch_num must be >= 1")
    if args.num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    data_dir = Path(args.data_dir)
    circuit_dir = _find_circuit_dir(
        data_dir,
        code=args.code,
        d=args.d,
        rounds=str(args.rounds),
        noise_model=args.noise_model,
        p=args.p,
        xyz=args.xyz,
        ibm_reproduce=args.ibm_reproduce,
    )

    dem_path = circuit_dir / "dem.dem"
    sampled_data_dir = circuit_dir / "sampled_data"
    dets_path = sampled_data_dir / f"det_batch={args.batch_num}.b8"
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    if not dets_path.exists():
        raise FileNotFoundError(f"Batch file not found: {dets_path}")

    decoder_config_path = Path(args.decoder_config)
    decoder_factory, decoder_info = load_decoder_factory(decoder_config_path, dem_path)
    if decoder_info.metric_name != FORCING_DEGRADATION_METRIC:
        raise ValueError(
            f"forcing_degradation_collect requires metric={FORCING_DEGRADATION_METRIC}, "
            f"got {decoder_info.metric_name}"
        )
    if decoder_info.decoder_name == "ILP":
        raise ValueError("forcing_degradation_test does not support ILP")

    options_suffix = _format_metric_options(decoder_info.metric_options)
    output_dir = (
        circuit_dir
        / "decoding_result"
        / f"decoder={decoder_info.decoder_name},metric={decoder_info.metric_name}{options_suffix}"
    )
    chunk_dir = output_dir / "chunks" / f"batch={args.batch_num}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    exec_config = ExecutionConfig(
        dem_path=dem_path,
        sampled_data_dir=sampled_data_dir,
        output_dir=chunk_dir,
        decoder_factory=decoder_factory,
        num_workers=args.num_workers,
        dets_paths=(dets_path,),
        verbose=bool(args.verbose),
        max_chunk_files=args.max_chunk_files,
        merge_chunk_group_size=args.merge_chunk_group_size,
    )

    start_time = datetime.now(timezone.utc)
    outcome = run_manager(exec_config)
    metrics_recorded = (
        _collect_forcing_degradation_results(chunk_dir, output_dir, args.batch_num)
        if outcome.results
        else []
    )
    if args.cleanup_intermediate:
        cleanup_intermediate(chunk_dir)

    incomplete_path = output_dir / INCOMPLETE_SHOTS_FILENAME
    if outcome.incomplete:
        write_incomplete_shots(incomplete_path, outcome.incomplete)
    elif incomplete_path.exists():
        incomplete_path.unlink()

    end_time = datetime.now(timezone.utc)
    metadata = build_decoding_metadata(
        decoder_info=decoder_info,
        dem_path=dem_path,
        dets_paths=(dets_path,),
        num_workers=args.num_workers,
        start_time=start_time,
        end_time=end_time,
        results=outcome.results,
        metrics_recorded=metrics_recorded,
        incomplete_ranges=outcome.incomplete,
    )
    write_metadata(metadata_file_path(output_dir, args.batch_num), metadata)

    if outcome.incomplete:
        total = sum(r.shot_id_end - r.shot_id_start for r in outcome.incomplete)
        raise IncompleteTasksError(
            f"{len(outcome.incomplete)} task(s) covering {total} shot(s) did not "
            f"complete. Outputs for completed shots were written to {output_dir}. "
            f"See {incomplete_path} for affected shot_id ranges."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
