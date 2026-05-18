from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
import os
import sys
import time
import resource
from typing import Iterable

import polars as pl
import stim

from decoder_confidence.execution.batching import estimate_shots_per_task
from decoder_confidence.execution.models import (
    DecoderFactory,
    SimulationTask,
    WorkerConfig,
    WorkerResult,
)
from decoder_confidence.execution.worker import init_worker, run_task


@dataclass(frozen=True)
class ExecutionConfig:
    dem_path: Path
    sampled_data_dir: Path
    output_dir: Path
    decoder_factory: DecoderFactory
    num_workers: int
    initial_probe_shots: int = 1000
    target_task_seconds: float = 30.0
    max_shots_per_task: int | None = None
    core_ids: tuple[int, ...] | None = None
    dets_paths: tuple[Path, ...] | None = None
    verbose: bool = False
    max_chunk_files: int | None = 1000
    merge_chunk_group_size: int | None = None


@dataclass(frozen=True)
class DetsFileInfo:
    path: Path
    total_shots: int
    shot_id_offset: int


def _get_dem_counts(dem_path: Path) -> tuple[int, int]:
    dem = stim.DetectorErrorModel.from_file(str(dem_path))
    try:
        return int(dem.num_detectors), int(dem.num_observables)
    except AttributeError as exc:
        raise ValueError(
            "stim.DetectorErrorModel missing num_detectors/num_observables"
        ) from exc


def _bytes_per_shot(num_detectors: int, num_observables: int) -> int:
    return (num_detectors + num_observables + 7) // 8


def _discover_dets_files(sampled_data_dir: Path) -> list[Path]:
    return sorted(sampled_data_dir.glob("*.b8"))


def _build_file_infos(
    dets_paths: Iterable[Path], bytes_per_shot: int
) -> list[DetsFileInfo]:
    infos: list[DetsFileInfo] = []
    offset = 0
    for path in dets_paths:
        size = path.stat().st_size
        if size % bytes_per_shot != 0:
            raise ValueError(
                f"File size for {path} not aligned to bytes_per_shot={bytes_per_shot}"
            )
        total_shots = size // bytes_per_shot
        infos.append(DetsFileInfo(path=path, total_shots=total_shots, shot_id_offset=offset))
        offset += total_shots
    return infos


def _select_core_ids(num_workers: int, core_ids: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if core_ids:
        return core_ids
    try:
        available = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None
    if not available:
        return None
    return tuple(available[:num_workers])


def _rss_mb() -> float:
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage_kb / (1024.0 * 1024.0)
    return usage_kb / 1024.0


def _resolve_merge_group_size(
    max_chunk_files: int | None, merge_chunk_group_size: int | None
) -> int | None:
    if max_chunk_files is None or max_chunk_files <= 0:
        return None
    if merge_chunk_group_size is None or merge_chunk_group_size <= 0:
        return min(200, max_chunk_files)
    return merge_chunk_group_size


def _next_part_index(part_dir: Path) -> int:
    if not part_dir.exists():
        return 1
    max_index = 0
    for path in part_dir.glob("part_*.parquet"):
        stem = path.stem
        if "_" not in stem:
            continue
        suffix = stem.split("_", 1)[1]
        try:
            max_index = max(max_index, int(suffix))
        except ValueError:
            continue
    return max_index + 1


def _merge_chunks_into_part(
    chunk_paths: list[Path], part_dir: Path, part_index: int
) -> Path:
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"part_{part_index:06d}.parquet"
    scan = pl.scan_parquet([str(path) for path in chunk_paths])
    scan.sink_parquet(part_path, compression="zstd")
    for path in chunk_paths:
        try:
            path.unlink()
        except OSError:
            pass
    return part_path


def _run_probe(worker_config: WorkerConfig, probe_task: SimulationTask) -> WorkerResult:
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1, initializer=init_worker, initargs=(worker_config,)) as pool:
        return pool.apply(run_task, (probe_task,))


def run_manager(config: ExecutionConfig) -> list[WorkerResult]:
    if config.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1 but got {config.num_workers}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    num_detectors, num_observables = _get_dem_counts(config.dem_path)
    bytes_per_shot = _bytes_per_shot(num_detectors, num_observables)

    dets_paths = (
        list(config.dets_paths)
        if config.dets_paths is not None
        else _discover_dets_files(config.sampled_data_dir)
    )
    if not dets_paths:
        raise FileNotFoundError(
            f"No .b8 files found under {config.sampled_data_dir}"
        )

    file_infos = _build_file_infos(dets_paths, bytes_per_shot)

    probe_info = file_infos[0]
    probe_shots = min(config.initial_probe_shots, probe_info.total_shots)
    if probe_shots <= 0:
        raise ValueError("Probe shots must be > 0")

    core_ids = _select_core_ids(config.num_workers, config.core_ids)
    worker_config = WorkerConfig(
        dem_path=config.dem_path,
        output_dir=config.output_dir,
        decoder_factory=config.decoder_factory,
        core_ids=core_ids,
    )

    probe_task = SimulationTask(
        dets_path=probe_info.path,
        start_shot_index=0,
        num_shots=probe_shots,
        batch_id=0,
        shot_id_offset=probe_info.shot_id_offset,
    )

    probe_result = _run_probe(worker_config, probe_task)
    if probe_result.status != "ok":
        raise RuntimeError(f"Probe task failed: {probe_result.message}")

    if probe_result.output_path.exists():
        try:
            probe_result.output_path.unlink()
        except OSError:
            pass

    shots_per_task = estimate_shots_per_task(
        probe_shots,
        probe_result.duration_s,
        config.target_task_seconds,
        min_shots=1,
        max_shots=config.max_shots_per_task,
    )

    tasks: list[SimulationTask] = []
    batch_id = 1
    for info in file_infos:
        start = 0
        while start < info.total_shots:
            num = min(shots_per_task, info.total_shots - start)
            tasks.append(
                SimulationTask(
                    dets_path=info.path,
                    start_shot_index=start,
                    num_shots=num,
                    batch_id=batch_id,
                    shot_id_offset=info.shot_id_offset,
                )
            )
            batch_id += 1
            start += num

    results: list[WorkerResult] = []
    start_time = time.perf_counter()
    total_tasks = len(tasks)
    completed_tasks = 0
    ok_count = 0
    skipped_count = 0
    error_count = 0
    ok_duration_sum = 0.0

    max_chunk_files = config.max_chunk_files
    merge_group_size = _resolve_merge_group_size(
        max_chunk_files, config.merge_chunk_group_size
    )
    part_dir = config.output_dir / "parts"
    part_index = _next_part_index(part_dir)
    pending_chunks: list[Path] = []
    seen_chunks: set[Path] = set()

    def _enqueue_chunk(path: Path) -> None:
        if path in seen_chunks:
            return
        seen_chunks.add(path)
        pending_chunks.append(path)

    def _maybe_merge_chunks() -> None:
        nonlocal part_index
        if max_chunk_files is None or max_chunk_files <= 0:
            return
        if merge_group_size is None or merge_group_size <= 0:
            return
        while len(pending_chunks) >= max_chunk_files:
            group_size = min(merge_group_size, len(pending_chunks))
            group = [path for path in pending_chunks[:group_size] if path.exists()]
            del pending_chunks[:group_size]
            if not group:
                continue
            try:
                part_path = _merge_chunks_into_part(group, part_dir, part_index)
            except Exception as exc:
                pending_chunks[:0] = group
                if config.verbose:
                    print(f"merge failed: {exc}")
                break
            if config.verbose:
                print(
                    f"merged {len(group)} chunks -> {part_path}"
                )
            part_index += 1

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=config.num_workers, initializer=init_worker, initargs=(worker_config,)
    ) as pool:
        for result in pool.imap_unordered(run_task, tasks, chunksize=1):
            results.append(result)
            completed_tasks += 1
            if result.status == "ok":
                ok_count += 1
                ok_duration_sum += result.duration_s
            elif result.status == "skipped":
                skipped_count += 1
            else:
                error_count += 1

            if result.status in {"ok", "skipped"}:
                _enqueue_chunk(result.output_path)
                _maybe_merge_chunks()

            if config.verbose:
                elapsed = time.perf_counter() - start_time
                avg_duration = ok_duration_sum / ok_count if ok_count else None
                avg_str = f"{avg_duration:.2f}s" if avg_duration is not None else "-"
                rss_mb = _rss_mb()
                print(
                    "progress "
                    f"{completed_tasks}/{total_tasks} "
                    f"ok={ok_count} skip={skipped_count} err={error_count} "
                    f"last={result.duration_s:.2f}s avg={avg_str} "
                    f"rss={rss_mb:.1f}MB elapsed={elapsed:.1f}s"
                )

    errors = [result for result in results if result.status == "error"]
    if errors:
        messages = "\n".join(
            f"batch_id={result.batch_id} msg={result.message}" for result in errors
        )
        raise RuntimeError(f"One or more tasks failed:\n{messages}")

    return results
