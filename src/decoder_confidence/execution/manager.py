from __future__ import annotations

import multiprocessing as mp
import time

from decoder_confidence.execution._execution_utils import (
    build_file_infos,
    bytes_per_shot_count,
    discover_dets_files,
    get_dem_counts,
    merge_chunks_into_part,
    next_part_index,
    resolve_merge_group_size,
    rss_mb,
    select_core_ids,
)
from decoder_confidence.execution.batching import (
    estimate_shots_per_task,
    estimate_task_timeout_s,
)
from decoder_confidence.execution.models import (
    DetsFileInfo,
    ExecutionConfig,
    RunOutcome,
    SharedEnvDecoderFactory,
    SimulationTask,
    WorkerConfig,
    WorkerResult,
)
from decoder_confidence.execution.pool_runner import run_pool_tasks
from decoder_confidence.execution.worker import init_worker, run_task

# Re-exported for backward compatibility so existing imports keep working.
__all__ = [
    "ExecutionConfig",
    "DetsFileInfo",
    "run_manager",
]


def _run_manager_multiprocess(config: ExecutionConfig) -> RunOutcome:
    """Multiprocessing execution path for decoders without session constraints."""
    if config.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1 but got {config.num_workers}")

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

    probe_info = file_infos[0]
    probe_shots = min(config.initial_probe_shots, probe_info.total_shots)
    if probe_shots <= 0:
        raise ValueError("Probe shots must be > 0")

    core_ids = select_core_ids(config.num_workers, config.core_ids)
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

    start_time = time.perf_counter()
    completed_tasks = 0
    ok_count = 0
    skipped_count = 0
    error_count = 0
    ok_duration_sum = 0.0

    max_chunk_files = config.max_chunk_files
    merge_group_size = resolve_merge_group_size(max_chunk_files, config.merge_chunk_group_size)
    part_dir = config.output_dir / "parts"
    part_index = next_part_index(part_dir)
    pending_chunks: list = []
    seen_chunks: set = set()

    def _enqueue_chunk(path) -> None:
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

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=config.num_workers,
        initializer=init_worker,
        initargs=(worker_config,),
        maxtasksperchild=config.maxtasksperchild,
    ) as pool:
        # Run probe inside the already-initialised pool so workers are spawned
        # only once.  Using a separate Pool for the probe (as before) caused a
        # double spawn cycle: 1-worker pool for probe, then 96-worker pool for
        # the actual tasks — each cycle re-imports heavy libraries from NFS.
        probe_result = pool.apply(run_task, (probe_task,))
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
        task_timeout_s = estimate_task_timeout_s(
            probe_shots,
            probe_result.duration_s,
            shots_per_task,
            timeout_multiplier=config.timeout_multiplier,
            min_task_timeout_s=config.min_task_timeout_s,
        )
        if config.verbose:
            print(
                f"shots_per_task={shots_per_task} task_timeout_s={task_timeout_s:.1f}"
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

        total_tasks = len(tasks)

        def _on_result(result: WorkerResult) -> None:
            nonlocal completed_tasks, ok_count, skipped_count, error_count, ok_duration_sum
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
                print(
                    "progress "
                    f"{completed_tasks}/{total_tasks} "
                    f"ok={ok_count} skip={skipped_count} err={error_count} "
                    f"last={result.duration_s:.2f}s avg={avg_str} "
                    f"rss={rss_mb():.1f}MB elapsed={elapsed:.1f}s"
                )

        outcome = run_pool_tasks(
            tasks,
            pool=pool,
            ctx=ctx,
            worker_config=worker_config,
            num_workers=config.num_workers,
            maxtasksperchild=config.maxtasksperchild,
            task_timeout_s=task_timeout_s,
            max_retries=config.max_task_retries,
            on_result=_on_result,
        )

    return outcome


def run_manager(config: ExecutionConfig) -> RunOutcome:
    """Run decoding simulation tasks, routing to the appropriate executor.

    - ``SharedEnvDecoderFactory`` (e.g. ILP/Gurobi) → threaded executor
      (1 shared env = 1 WLS session, N concurrent solver threads)
    - All other factories → multiprocessing pool executor
    """
    if isinstance(config.decoder_factory, SharedEnvDecoderFactory):
        from decoder_confidence.execution.threaded_manager import run_manager_threaded
        return run_manager_threaded(config)
    return _run_manager_multiprocess(config)
