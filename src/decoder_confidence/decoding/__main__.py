from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from decoder_confidence.varint import write_obs_flip_idx_file
from decoder_confidence.decoding.decoder_factory import load_decoder_factory
from decoder_confidence.decoding.incomplete import (
    INCOMPLETE_SHOTS_FILENAME,
    write_incomplete_shots,
)
from decoder_confidence.decoding.metadata import build_decoding_metadata, write_metadata
from decoder_confidence.execution.manager import ExecutionConfig, run_manager
from decoder_confidence.execution.models import IncompleteTasksError


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "t", "yes", "y"}:
        return True
    if value_str in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"value must be a boolean value but got {value}")


def _parse_dir_name(name: str) -> dict[str, Any] | None:
    parts = name.split(",")
    values: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value

    rounds_value = values.get("rounds") or values.get("r")
    if not rounds_value:
        return None
    if "code" not in values or "d" not in values or "noisemodel" not in values or "p" not in values:
        return None

    try:
        d_value = int(values["d"])
    except ValueError:
        return None
    try:
        p_value = float(values["p"])
    except ValueError:
        return None

    xyz_value: bool | None = None
    if "xyz" in values:
        try:
            xyz_value = _parse_bool(values["xyz"])
        except ValueError:
            return None

    ibm_reproduce_value: bool | None = None
    if "ibm_reproduce" in values:
        try:
            ibm_reproduce_value = _parse_bool(values["ibm_reproduce"])
        except ValueError:
            return None

    return {
        "code": values["code"],
        "d": d_value,
        "rounds": rounds_value,
        "noisemodel": values["noisemodel"],
        "p": p_value,
        "xyz": xyz_value,
        "ibm_reproduce": ibm_reproduce_value,
    }


def _find_circuit_dir(
    data_dir: Path,
    *,
    code: str,
    d: int,
    rounds: str,
    noise_model: str,
    p: float,
    xyz: bool | None,
    ibm_reproduce: bool = False,
) -> Path:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    matches: list[Path] = []
    for path in data_dir.rglob("*"):
        if not path.is_dir():
            continue
        parsed = _parse_dir_name(path.name)
        if parsed is None:
            continue
        if parsed["code"] != code:
            continue
        if parsed["d"] != d:
            continue
        if parsed["rounds"] != rounds:
            continue
        if parsed["noisemodel"] != noise_model:
            continue
        if not math.isclose(parsed["p"], p, rel_tol=0.0, abs_tol=1e-12):
            continue
        if xyz is not None:
            if parsed.get("xyz") is None:
                continue
            if parsed["xyz"] != xyz:
                continue
        if ibm_reproduce:
            if parsed.get("ibm_reproduce") is not True:
                continue
        else:
            if parsed.get("ibm_reproduce") is True:
                continue
        matches.append(path)

    if not matches:
        expected = f"code={code},d={d},rounds={rounds},noisemodel={noise_model},p={p}"
        if xyz is not None:
            expected = f"{expected},xyz={xyz}"
        if ibm_reproduce:
            expected = f"{expected},ibm_reproduce=True"
        raise FileNotFoundError(
            f"Circuit directory not found under {data_dir}. Expected {expected}"
        )
    if len(matches) > 1:
        match_list = "\n".join(str(path) for path in matches)
        raise FileExistsError(
            "Multiple circuit directories matched the requested parameters:\n" + match_list
        )

    return matches[0]


def _collect_results(
    chunk_dir: Path,
    output_dir: Path,
    batch_num: int,
    metric: str,
) -> list[str]:
    chunk_paths = sorted(chunk_dir.glob("chunk_*.parquet"))
    part_paths = sorted((chunk_dir / "parts").glob("part_*.parquet"))
    input_paths = chunk_paths + part_paths
    if not input_paths:
        raise FileNotFoundError(
            f"No chunk/part parquet files found in {chunk_dir}"
        )

    scan = pl.scan_parquet([str(path) for path in input_paths])

    logicalerror_path = output_dir / f"logicalerror_batch={batch_num}.parquet"
    scan.select(["shot_id", "is_logical_error"]).sink_parquet(
        logicalerror_path, compression="zstd"
    )

    schema = scan.collect_schema()
    metric_cols = [name for name in schema if name.startswith("metric_")]
    if not metric_cols:
        raise ValueError("No metric columns found in chunk outputs")

    metric_names = sorted({name[len("metric_") :] for name in metric_cols})
    if metric not in metric_names:
        available = ", ".join(metric_names)
        raise ValueError(
            f"Configured metric '{metric}' missing in chunk outputs (found: {available})"
        )

    for metric_name in metric_names:
        metric_col = f"metric_{metric_name}"
        metric_path = output_dir / f"metric={metric_name}_batch={batch_num}.parquet"
        scan.select(["shot_id", pl.col(metric_col).alias(metric_name)]).sink_parquet(
            metric_path, compression="zstd"
        )

    if "obs_flip_idx" in schema:
        obs_df = scan.select(["shot_id", "obs_flip_idx"]).sort("shot_id").collect()
        blobs: list[bytes] = obs_df["obs_flip_idx"].to_list()
        obs_path = output_dir / f"obs_flip_idx_batch={batch_num}.bin"
        write_obs_flip_idx_file(obs_path, blobs)

    return metric_names


def _cleanup_intermediate(chunk_dir: Path) -> None:
    for path in chunk_dir.glob("chunk_*.parquet"):
        try:
            path.unlink()
        except OSError:
            pass

    part_dir = chunk_dir / "parts"
    if part_dir.exists():
        for path in part_dir.glob("part_*.parquet"):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            part_dir.rmdir()
        except OSError:
            pass


def _format_metric_options(metric_options: Mapping[str, Any]) -> str:
    if not metric_options:
        return ""
    parts = [f"{key}={repr(metric_options[key])}" for key in sorted(metric_options)]
    return "," + ",".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run step3 decoding for a specific batch."
    )
    parser.add_argument("--code", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--noise_model", required=True)
    parser.add_argument("--rounds", required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--batch_num", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--decoder_config", required=True)
    parser.add_argument(
        "--xyz",
        required=True,
        type=_parse_bool,
        help="Match circuit directories with xyz=<value>",
    )
    parser.add_argument(
        "--ibm_reproduce",
        action="store_true",
        default=False,
        help="If set, use circuit directories with ibm_reproduce=True",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        type=_parse_bool,
        help="If true, print progress and resource usage during decoding",
    )
    parser.add_argument(
        "--max_chunk_files",
        default=1000,
        type=int,
        help="Merge chunk files into parts once this count is exceeded (<=0 disables)",
    )
    parser.add_argument(
        "--merge_chunk_group_size",
        default=None,
        type=int,
        help="Number of chunks to merge per part file (defaults to min(200, max_chunk_files))",
    )
    parser.add_argument(
        "--cleanup_intermediate",
        default=True,
        type=_parse_bool,
        help="If true, remove chunk/part parquet files after final outputs are written",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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

    if decoder_info.decoder_name == "ILP":
        num_threads = int(decoder_info.decoder_options.get("threads") or 1)
        try:
            available_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cores = os.cpu_count() or 1
        if num_threads * args.num_workers > available_cores:
            raise ValueError(
                f"num_threads ({num_threads}) * num_workers ({args.num_workers}) = "
                f"{num_threads * args.num_workers} exceeds available cores ({available_cores}). "
                "Reduce threads in decoder_options or num_workers."
            )

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
    metric_names = (
        _collect_results(chunk_dir, output_dir, args.batch_num, decoder_info.metric_name)
        if outcome.results
        else []
    )
    if args.cleanup_intermediate:
        _cleanup_intermediate(chunk_dir)

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
        metrics_recorded=metric_names,
        incomplete_ranges=outcome.incomplete,
    )
    write_metadata(output_dir / "metadata.json", metadata)

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
