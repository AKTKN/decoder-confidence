"""Threaded execution manager for session-limited decoders (e.g. Gurobi ILP).

Uses a single shared environment (1 WLS session) and a ThreadPoolExecutor so
that Gurobi's C-layer solves run in parallel while only one session is consumed.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import stim

from decoder_confidence.execution._execution_utils import (
    build_file_infos,
    bytes_per_shot_count,
    discover_dets_files,
    get_dem_counts,
    merge_chunks_into_part,
    next_part_index,
    resolve_merge_group_size,
    rss_mb,
)
from decoder_confidence.execution.batching import estimate_shots_per_task
from decoder_confidence.execution.hashing import chunk_path
from decoder_confidence.execution.models import (
    DecoderBase,
    ExecutionConfig,
    SharedEnvDecoderFactory,
    SimulationTask,
    WorkerResult,
)
from decoder_confidence.varint import encode_obs_flip_shot
from decoder_confidence.execution.worker import (
    _normalize_metrics,
    _normalize_predictions,
    read_b8_slice,
)


def _run_task(
    task: SimulationTask,
    *,
    decoder: DecoderBase,
    num_detectors: int,
    num_observables: int,
    bytes_per_shot: int,
    decoder_name: str,
    output_dir: Path,
) -> WorkerResult:
    """Execute a single simulation task with the supplied decoder instance."""
    output_path = chunk_path(output_dir, task, decoder_name)
    if output_path.exists():
        return WorkerResult(
            status="skipped",
            duration_s=0.0,
            output_path=output_path,
            num_shots=task.num_shots,
            batch_id=task.batch_id,
        )

    start = time.perf_counter()
    status = "ok"
    message: str | None = None
    try:
        data = read_b8_slice(
            task.dets_path,
            task.start_shot_index,
            task.num_shots,
            bytes_per_shot=bytes_per_shot,
            num_detectors=num_detectors,
            num_observables=num_observables,
        )
        dets = data[:, :num_detectors]
        obs = data[:, num_detectors:]

        result = decoder.decode(dets)
        predictions = _normalize_predictions(result.predictions, task.num_shots, num_observables)

        if num_observables > 0:
            mismatch = predictions ^ obs
            is_logical_error = mismatch.any(axis=1)
        else:
            is_logical_error = np.zeros(task.num_shots, dtype=bool)

        shot_ids = (
            task.shot_id_offset
            + task.start_shot_index
            + np.arange(task.num_shots, dtype=np.int64)
        )
        columns: dict[str, Any] = {
            "shot_id": shot_ids,
            "is_logical_error": is_logical_error,
        }
        columns.update(_normalize_metrics(result.metrics, task.num_shots))

        if result.obs_flip_idx is not None:
            blobs = [encode_obs_flip_shot(idxs) for idxs in result.obs_flip_idx]
            columns["obs_flip_idx"] = pl.Series("obs_flip_idx", blobs, dtype=pl.Binary)

        df = pl.DataFrame(columns)
        df.write_parquet(output_path, compression="zstd")
    except Exception as exc:
        status = "error"
        message = str(exc)
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

    return WorkerResult(
        status=status,
        duration_s=time.perf_counter() - start,
        output_path=output_path,
        num_shots=task.num_shots,
        batch_id=task.batch_id,
        message=message,
    )


def run_manager_threaded(config: ExecutionConfig) -> list[WorkerResult]:
    """Run decoding tasks with a thread pool sharing one Gurobi env.

    Called automatically by ``run_manager`` when the decoder factory is a
    ``SharedEnvDecoderFactory``.  Worker threads each hold a thread-local
    decoder instance but all share the single env created here, so exactly
    one WLS session is consumed for the lifetime of the simulation.
    """
    factory = config.decoder_factory
    if not isinstance(factory, SharedEnvDecoderFactory):
        raise TypeError(
            f"run_manager_threaded requires a SharedEnvDecoderFactory, got {type(factory)}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    num_detectors, num_observables = get_dem_counts(config.dem_path)
    bps = bytes_per_shot_count(num_detectors, num_observables)

    dets_paths = (
        list(config.dets_paths)
        if config.dets_paths is not None
        else discover_dets_files(config.sampled_data_dir)
    )
    if not dets_paths:
        raise FileNotFoundError(f"No .b8 files found under {config.sampled_data_dir}")

    file_infos = build_file_infos(dets_paths, bps)

    # Load DEM once — read-only, safe to share across threads.
    dem = stim.DetectorErrorModel.from_file(str(config.dem_path))

    # Build the shared env exactly once (= 1 WLS session).
    shared_env = factory.build_env()

    # Probe: run a small batch inline to calibrate shots_per_task.
    probe_info = file_infos[0]
    probe_shots = min(config.initial_probe_shots, probe_info.total_shots)
    if probe_shots <= 0:
        raise ValueError("Probe shots must be > 0")

    probe_decoder = factory.build_decoder(shared_env, dem)
    decoder_name = probe_decoder.__class__.__name__

    probe_task = SimulationTask(
        dets_path=probe_info.path,
        start_shot_index=0,
        num_shots=probe_shots,
        batch_id=0,
        shot_id_offset=probe_info.shot_id_offset,
    )
    probe_result = _run_task(
        probe_task,
        decoder=probe_decoder,
        num_detectors=num_detectors,
        num_observables=num_observables,
        bytes_per_shot=bps,
        decoder_name=decoder_name,
        output_dir=config.output_dir,
    )
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

    # Build task list.
    tasks: list[SimulationTask] = []
    batch_id = 1
    for info in file_infos:
        start_idx = 0
        while start_idx < info.total_shots:
            num = min(shots_per_task, info.total_shots - start_idx)
            tasks.append(
                SimulationTask(
                    dets_path=info.path,
                    start_shot_index=start_idx,
                    num_shots=num,
                    batch_id=batch_id,
                    shot_id_offset=info.shot_id_offset,
                )
            )
            batch_id += 1
            start_idx += num

    # Thread-local decoder registry: each thread builds its own ILPDecoder
    # using the shared env so only one WLS session is consumed.
    _tl: threading.local = threading.local()

    def _get_decoder() -> DecoderBase:
        if not hasattr(_tl, "decoder"):
            _tl.decoder = factory.build_decoder(shared_env, dem)
        return _tl.decoder

    def _process_task(task: SimulationTask) -> WorkerResult:
        return _run_task(
            task,
            decoder=_get_decoder(),
            num_detectors=num_detectors,
            num_observables=num_observables,
            bytes_per_shot=bps,
            decoder_name=decoder_name,
            output_dir=config.output_dir,
        )

    # Progress tracking and chunk-merge — mirrors manager.py logic.
    results: list[WorkerResult] = []
    wall_start = time.perf_counter()
    total_tasks = len(tasks)
    ok_count = skipped_count = error_count = 0
    ok_duration_sum = 0.0

    max_chunk_files = config.max_chunk_files
    merge_group_size = resolve_merge_group_size(max_chunk_files, config.merge_chunk_group_size)
    part_dir = config.output_dir / "parts"
    part_index = next_part_index(part_dir)
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
            group = [p for p in pending_chunks[:group_size] if p.exists()]
            del pending_chunks[:group_size]
            if not group:
                continue
            try:
                part_path = merge_chunks_into_part(group, part_dir, part_index)
            except Exception as exc:
                pending_chunks[:0] = group
                if config.verbose:
                    print(f"merge failed: {exc}")
                break
            if config.verbose:
                print(f"merged {len(group)} chunks -> {part_path}")
            part_index += 1

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {executor.submit(_process_task, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed = len(results)

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
                elapsed = time.perf_counter() - wall_start
                avg_duration = ok_duration_sum / ok_count if ok_count else None
                avg_str = f"{avg_duration:.2f}s" if avg_duration is not None else "-"
                print(
                    f"progress {completed}/{total_tasks} "
                    f"ok={ok_count} skip={skipped_count} err={error_count} "
                    f"last={result.duration_s:.2f}s avg={avg_str} "
                    f"rss={rss_mb():.1f}MB elapsed={elapsed:.1f}s"
                )

    errors = [r for r in results if r.status == "error"]
    if errors:
        messages = "\n".join(
            f"batch_id={r.batch_id} msg={r.message}" for r in errors
        )
        raise RuntimeError(f"One or more tasks failed:\n{messages}")

    return results
