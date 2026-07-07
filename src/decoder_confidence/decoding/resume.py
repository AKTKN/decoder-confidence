"""Resume an interrupted Step 3 (collect) decoding run.

Scans existing chunk/part parquets in a partially-completed output directory,
identifies missing shot ranges, and resumes decoding for only those shots.
After all shots are complete, runs the normal collect/finalize pipeline.

Usage:
    python -m decoder_confidence.decoding.resume \\
        --output_dir path/to/decoding_result/decoder=ILP,metric=logical_gap \\
        --decoder_config path/to/step3_config.yaml \\
        --batch_num 1 \\
        --num_workers 96
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl
import stim

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
from decoder_confidence.decoding.result_collection import (
    cleanup_intermediate,
    collect_results,
)
from decoder_confidence.execution._execution_utils import (
    bytes_per_shot_count,
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
    IncompleteRange,
    IncompleteTasksError,
    RunOutcome,
    SharedEnvDecoderFactory,
    SimulationTask,
    WorkerConfig,
    WorkerResult,
)
from decoder_confidence.execution.pool_runner import run_pool_tasks
from decoder_confidence.execution.threaded_manager import _run_task as _threaded_run_task
from decoder_confidence.execution.worker import init_worker, run_task


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean but got {value!r}")


def _detect_batch_nums(output_dir: Path) -> list[int]:
    chunks_dir = output_dir / "chunks"
    if not chunks_dir.exists():
        return []
    nums: list[int] = []
    for d in chunks_dir.iterdir():
        if d.is_dir() and d.name.startswith("batch="):
            try:
                nums.append(int(d.name.split("=", 1)[1]))
            except (ValueError, IndexError):
                pass
    return sorted(nums)


# ── Coverage analysis ─────────────────────────────────────────────────────────

def _scan_covered_ranges(chunk_dir: Path) -> list[tuple[int, int]]:
    """Return sorted, merged (start_shot_id, end_exclusive) ranges from existing parquets."""
    chunk_paths = sorted(chunk_dir.glob("chunk_*.parquet"))
    part_paths = sorted((chunk_dir / "parts").glob("part_*.parquet"))
    all_paths = chunk_paths + part_paths
    if not all_paths:
        return []

    raw: list[tuple[int, int]] = []
    for path in all_paths:
        try:
            stats = (
                pl.scan_parquet(str(path))
                .select([
                    pl.col("shot_id").min().alias("lo"),
                    pl.col("shot_id").max().alias("hi"),
                    pl.len().alias("cnt"),
                ])
                .collect()
            )
            lo = int(stats["lo"][0])
            hi = int(stats["hi"][0])
            cnt = int(stats["cnt"][0])
            if cnt == hi - lo + 1:
                raw.append((lo, hi + 1))
            else:
                # Non-contiguous block (e.g. merged part spanning gaps): walk sub-ranges
                ids = (
                    pl.scan_parquet(str(path))
                    .select("shot_id")
                    .sort("shot_id")
                    .collect()["shot_id"]
                    .to_list()
                )
                seg_s = ids[0]
                prev = ids[0]
                for x in ids[1:]:
                    if x != prev + 1:
                        raw.append((seg_s, prev + 1))
                        seg_s = x
                    prev = x
                raw.append((seg_s, prev + 1))
        except Exception as exc:
            print(f"[resume] Warning: cannot read {path.name}: {exc}")

    raw.sort()
    merged: list[list[int]] = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _uncovered_ranges(
    covered: list[tuple[int, int]],
    shot_id_offset: int,
    total_shots: int,
) -> list[tuple[int, int]]:
    pos = shot_id_offset
    full_end = shot_id_offset + total_shots
    gaps: list[tuple[int, int]] = []
    for cs, ce in covered:
        if cs > pos:
            gaps.append((pos, cs))
        pos = max(pos, ce)
    if pos < full_end:
        gaps.append((pos, full_end))
    return gaps


def _make_resume_tasks(
    uncovered: list[tuple[int, int]],
    dets_path: Path,
    shot_id_offset: int,
    shots_per_task: int,
) -> list[SimulationTask]:
    tasks: list[SimulationTask] = []
    batch_id = 1
    for r_start, r_end in uncovered:
        pos = r_start - shot_id_offset
        end = r_end - shot_id_offset
        while pos < end:
            num = min(shots_per_task, end - pos)
            tasks.append(SimulationTask(
                dets_path=dets_path,
                start_shot_index=pos,
                num_shots=num,
                batch_id=batch_id,
                shot_id_offset=shot_id_offset,
            ))
            batch_id += 1
            pos += num
    return tasks


# ── Task execution ─────────────────────────────────────────────────────────────

def _chunk_loop(
    result_iter: Iterable[tuple[SimulationTask, WorkerResult]],
    *,
    total_tasks: int,
    chunk_dir: Path,
    verbose: bool,
    max_chunk_files: int | None,
    merge_chunk_group_size: int | None,
) -> RunOutcome:
    """Consume (task, result) pairs, track progress, merge chunks into parts.

    Used by the threaded resume path only (Priority 2: a failed task is
    recorded as an ``IncompleteRange`` after a single attempt, with no
    retry/timeout -- see ``threaded_manager.run_manager_threaded`` for why
    Priority 3 does not apply to ``ThreadPoolExecutor``-based execution).
    """
    merge_group_size = resolve_merge_group_size(max_chunk_files, merge_chunk_group_size)
    part_dir = chunk_dir / "parts"
    part_index = next_part_index(part_dir)
    pending: list[Path] = []
    seen: set[Path] = set()
    results: list[WorkerResult] = []
    incomplete: list[IncompleteRange] = []
    completed = 0
    ok_count = skipped_count = error_count = 0
    ok_dur_sum = 0.0
    wall_start = time.perf_counter()

    def _enqueue(path: Path) -> None:
        nonlocal part_index
        if path in seen:
            return
        seen.add(path)
        pending.append(path)
        if not max_chunk_files or not merge_group_size:
            return
        while len(pending) >= max_chunk_files:
            gsz = min(merge_group_size, len(pending))
            group = [p for p in pending[:gsz] if p.exists()]
            del pending[:gsz]
            if not group:
                continue
            try:
                pp = merge_chunks_into_part(group, part_dir, part_index)
                if verbose:
                    print(f"[resume] merged {len(group)} chunks -> {pp.name}")
                part_index += 1
            except Exception as exc:
                pending[:0] = group
                if verbose:
                    print(f"[resume] merge failed: {exc}")
                break

    for task, result in result_iter:
        completed += 1
        if result.status == "ok":
            ok_count += 1
            ok_dur_sum += result.duration_s
            results.append(result)
        elif result.status == "skipped":
            skipped_count += 1
            results.append(result)
        else:
            error_count += 1
            incomplete.append(
                IncompleteRange.from_task(task, reason="error", message=result.message)
            )
        if result.status in {"ok", "skipped"}:
            _enqueue(result.output_path)
        if verbose:
            elapsed = time.perf_counter() - wall_start
            avg = ok_dur_sum / ok_count if ok_count else None
            avg_s = f"{avg:.2f}s" if avg else "-"
            print(
                f"[resume] {completed}/{total_tasks} "
                f"ok={ok_count} skip={skipped_count} err={error_count} "
                f"last={result.duration_s:.2f}s avg={avg_s} "
                f"rss={rss_mb():.1f}MB elapsed={elapsed:.1f}s"
            )

    return RunOutcome(results=results, incomplete=incomplete)




def _run_tasks_threaded(
    tasks: list[SimulationTask],
    *,
    dem: stim.DetectorErrorModel,
    shared_env,
    decoder_factory: SharedEnvDecoderFactory,
    num_detectors: int,
    num_observables: int,
    bps: int,
    decoder_name: str,
    chunk_dir: Path,
    num_workers: int,
    verbose: bool,
    max_chunk_files: int | None,
    merge_chunk_group_size: int | None,
) -> RunOutcome:
    tl = threading.local()

    def _get_decoder():
        if not hasattr(tl, "dec"):
            tl.dec = decoder_factory.build_decoder(shared_env, dem)
        return tl.dec

    def _process(task: SimulationTask) -> WorkerResult:
        return _threaded_run_task(
            task,
            decoder=_get_decoder(),
            num_detectors=num_detectors,
            num_observables=num_observables,
            bytes_per_shot=bps,
            decoder_name=decoder_name,
            output_dir=chunk_dir,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process, t): t for t in tasks}

        def _iter_results():
            for future in as_completed(futures):
                yield futures[future], future.result()

        return _chunk_loop(
            _iter_results(),
            total_tasks=len(tasks),
            chunk_dir=chunk_dir,
            verbose=verbose,
            max_chunk_files=max_chunk_files,
            merge_chunk_group_size=merge_chunk_group_size,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume an interrupted Step 3 (collect) decoding run."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the interrupted decoding_result/decoder=X,metric=Y directory.",
    )
    parser.add_argument(
        "--decoder_config",
        required=True,
        help="Path to the decoder YAML config (same file used in the original run).",
    )
    parser.add_argument(
        "--batch_num",
        type=int,
        default=None,
        help="Batch number to resume. Auto-detected if only one batch directory exists.",
    )
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--verbose", default=False, type=_parse_bool)
    parser.add_argument("--max_chunk_files", default=1000, type=int)
    parser.add_argument("--merge_chunk_group_size", default=None, type=int)
    parser.add_argument(
        "--cleanup_intermediate",
        default=True,
        type=_parse_bool,
        help="Remove chunk/part files after final outputs are written.",
    )
    parser.add_argument(
        "--timeout_multiplier",
        default=10.0,
        type=float,
        help="Per-task timeout = timeout_multiplier * probe-estimated task duration.",
    )
    parser.add_argument(
        "--min_task_timeout_s",
        default=60.0,
        type=float,
        help="Floor for the adaptive per-task timeout, in seconds.",
    )
    parser.add_argument(
        "--max_task_retries",
        default=1,
        type=int,
        help="Number of retries for a task that times out or errors.",
    )
    parser.add_argument(
        "--maxtasksperchild",
        default=50,
        type=int,
        help="Recycle each pool worker after this many tasks (<=0 disables).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    decoder_config_path = Path(args.decoder_config).resolve()
    num_workers = args.num_workers
    verbose = args.verbose
    max_chunk_files = args.max_chunk_files if args.max_chunk_files > 0 else None
    merge_chunk_group_size = args.merge_chunk_group_size
    maxtasksperchild = args.maxtasksperchild if args.maxtasksperchild > 0 else None

    if not output_dir.exists():
        raise FileNotFoundError(f"output_dir not found: {output_dir}")

    # Determine batch_num
    if args.batch_num is not None:
        batch_num = args.batch_num
    else:
        batch_nums = _detect_batch_nums(output_dir)
        if not batch_nums:
            raise FileNotFoundError(
                f"No chunks/batch=N directories found under {output_dir}. "
                "Specify --batch_num explicitly."
            )
        if len(batch_nums) > 1:
            raise ValueError(
                f"Multiple batches found: {batch_nums}. "
                "Specify --batch_num explicitly."
            )
        batch_num = batch_nums[0]
        print(f"[resume] Auto-detected batch_num={batch_num}")

    # Derive paths from output_dir hierarchy:
    #   circuit_dir/decoding_result/decoder=X,metric=Y/  ← output_dir
    circuit_dir = output_dir.parent.parent
    dem_path = circuit_dir / "dem.dem"
    sampled_data_dir = circuit_dir / "sampled_data"
    dets_path = sampled_data_dir / f"det_batch={batch_num}.b8"
    chunk_dir = output_dir / "chunks" / f"batch={batch_num}"

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    if not dets_path.exists():
        raise FileNotFoundError(f"Batch file not found: {dets_path}")

    # Load decoder
    decoder_factory, decoder_info = load_decoder_factory(decoder_config_path, dem_path)

    # ILP thread-count sanity check
    if decoder_info.decoder_name == "ILP":
        num_threads = int(decoder_info.decoder_options.get("threads") or 1)
        try:
            available_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cores = os.cpu_count() or 1
        if num_threads * num_workers > available_cores:
            raise ValueError(
                f"num_threads ({num_threads}) * num_workers ({num_workers}) = "
                f"{num_threads * num_workers} exceeds available cores ({available_cores}). "
                "Reduce threads in decoder_options or num_workers."
            )

    # Total shots from dets file
    num_detectors, num_observables = get_dem_counts(dem_path)
    bps = bytes_per_shot_count(num_detectors, num_observables)
    dets_size = dets_path.stat().st_size
    if dets_size % bps != 0:
        raise ValueError(
            f"dets file size {dets_size} not aligned to bytes_per_shot={bps}"
        )
    total_shots = dets_size // bps
    shot_id_offset = 0  # single .b8 file per batch

    # Scan existing parquets
    chunk_dir.mkdir(parents=True, exist_ok=True)
    covered = _scan_covered_ranges(chunk_dir)
    covered_shots = sum(e - s for s, e in covered)
    remaining_shots = total_shots - covered_shots

    print(f"[resume] output_dir  : {output_dir}")
    print(f"[resume] batch_num   : {batch_num}")
    print(f"[resume] decoder     : {decoder_info.decoder_name}  metric: {decoder_info.metric_name}")
    print(f"[resume] total shots : {total_shots}")
    print(f"[resume] covered     : {covered_shots}  ({100 * covered_shots / total_shots:.1f}%)")
    print(f"[resume] remaining   : {remaining_shots}")

    new_outcome: RunOutcome = RunOutcome([], [])

    if remaining_shots > 0:
        uncovered = _uncovered_ranges(covered, shot_id_offset, total_shots)

        # Probe from the start of the first uncovered range to estimate shots_per_task
        probe_start = uncovered[0][0] - shot_id_offset
        probe_shots = min(20, uncovered[0][1] - uncovered[0][0])

        is_threaded = isinstance(decoder_factory, SharedEnvDecoderFactory)

        if is_threaded:
            dem = stim.DetectorErrorModel.from_file(str(dem_path))
            shared_env = decoder_factory.build_env()
            probe_decoder = decoder_factory.build_decoder(shared_env, dem)
            decoder_name = probe_decoder.__class__.__name__

            probe_task = SimulationTask(
                dets_path=dets_path,
                start_shot_index=probe_start,
                num_shots=probe_shots,
                batch_id=0,
                shot_id_offset=shot_id_offset,
            )
            probe_result = _threaded_run_task(
                probe_task,
                decoder=probe_decoder,
                num_detectors=num_detectors,
                num_observables=num_observables,
                bytes_per_shot=bps,
                decoder_name=decoder_name,
                output_dir=chunk_dir,
            )
            if probe_result.status != "ok":
                raise RuntimeError(f"Probe failed: {probe_result.message}")
            if probe_result.output_path.exists():
                try:
                    probe_result.output_path.unlink()
                except OSError:
                    pass

            shots_per_task = estimate_shots_per_task(
                probe_shots, probe_result.duration_s, 30.0, min_shots=1,
            )
            if verbose:
                print(f"[resume] probe: {probe_shots} shots in {probe_result.duration_s:.3f}s "
                      f"→ shots_per_task={shots_per_task}")

            tasks = _make_resume_tasks(uncovered, dets_path, shot_id_offset, shots_per_task)
            print(f"[resume] {len(tasks)} tasks for {sum(t.num_shots for t in tasks)} remaining shots")

            start_time = datetime.now(timezone.utc)
            new_outcome = _run_tasks_threaded(
                tasks,
                dem=dem,
                shared_env=shared_env,
                decoder_factory=decoder_factory,
                num_detectors=num_detectors,
                num_observables=num_observables,
                bps=bps,
                decoder_name=decoder_name,
                chunk_dir=chunk_dir,
                num_workers=num_workers,
                verbose=verbose,
                max_chunk_files=max_chunk_files,
                merge_chunk_group_size=merge_chunk_group_size,
            )
        else:
            # Non-threaded path (e.g., CPLEX): run probe inside the main pool so
            # workers are spawned only once instead of twice (probe pool + task pool).
            core_ids = select_core_ids(num_workers, None)
            worker_config = WorkerConfig(
                dem_path=dem_path,
                output_dir=chunk_dir,
                decoder_factory=decoder_factory,
                core_ids=core_ids,
            )
            probe_task = SimulationTask(
                dets_path=dets_path,
                start_shot_index=probe_start,
                num_shots=probe_shots,
                batch_id=0,
                shot_id_offset=shot_id_offset,
            )
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=num_workers,
                initializer=init_worker,
                initargs=(worker_config,),
                maxtasksperchild=maxtasksperchild,
            ) as pool:
                probe_result = pool.apply(run_task, (probe_task,))
                if probe_result.status != "ok":
                    raise RuntimeError(f"Probe failed: {probe_result.message}")
                if probe_result.output_path.exists():
                    try:
                        probe_result.output_path.unlink()
                    except OSError:
                        pass

                shots_per_task = estimate_shots_per_task(
                    probe_shots, probe_result.duration_s, 30.0, min_shots=1,
                )
                task_timeout_s = estimate_task_timeout_s(
                    probe_shots,
                    probe_result.duration_s,
                    shots_per_task,
                    timeout_multiplier=args.timeout_multiplier,
                    min_task_timeout_s=args.min_task_timeout_s,
                )
                if verbose:
                    print(f"[resume] probe: {probe_shots} shots in {probe_result.duration_s:.3f}s "
                          f"→ shots_per_task={shots_per_task} task_timeout_s={task_timeout_s:.1f}")

                tasks = _make_resume_tasks(uncovered, dets_path, shot_id_offset, shots_per_task)
                print(f"[resume] {len(tasks)} tasks for {sum(t.num_shots for t in tasks)} remaining shots")

                start_time = datetime.now(timezone.utc)

                merge_group_size = resolve_merge_group_size(max_chunk_files, merge_chunk_group_size)
                part_dir = chunk_dir / "parts"
                part_index = next_part_index(part_dir)
                pending_chunks: list[Path] = []
                seen_chunks: set[Path] = set()
                ok_count = skipped_count = error_count = 0
                ok_dur_sum = 0.0
                completed = 0
                wall_start = time.perf_counter()
                total_tasks = len(tasks)

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
                            if verbose:
                                print(f"[resume] merge failed: {exc}")
                            break
                        if verbose:
                            print(f"[resume] merged {len(group)} chunks -> {part_path}")
                        part_index += 1

                def _on_result(result: WorkerResult) -> None:
                    nonlocal completed, ok_count, skipped_count, error_count, ok_dur_sum
                    completed += 1
                    if result.status == "ok":
                        ok_count += 1
                        ok_dur_sum += result.duration_s
                    elif result.status == "skipped":
                        skipped_count += 1
                    else:
                        error_count += 1

                    if result.status in {"ok", "skipped"}:
                        _enqueue_chunk(result.output_path)
                        _maybe_merge_chunks()

                    if verbose:
                        elapsed = time.perf_counter() - wall_start
                        avg = ok_dur_sum / ok_count if ok_count else None
                        avg_s = f"{avg:.2f}s" if avg is not None else "-"
                        print(
                            f"[resume] {completed}/{total_tasks} "
                            f"ok={ok_count} skip={skipped_count} err={error_count} "
                            f"last={result.duration_s:.2f}s avg={avg_s} "
                            f"rss={rss_mb():.1f}MB elapsed={elapsed:.1f}s"
                        )

                new_outcome = run_pool_tasks(
                    tasks,
                    pool=pool,
                    ctx=ctx,
                    worker_config=worker_config,
                    num_workers=num_workers,
                    maxtasksperchild=maxtasksperchild,
                    task_timeout_s=task_timeout_s,
                    max_retries=args.max_task_retries,
                    on_result=_on_result,
                    abort_on_error=True,
                )
    else:
        print("[resume] All shots already complete. Skipping decoding, running collection only.")
        start_time = datetime.now(timezone.utc)

    # Collect all chunks (original + newly produced) into final outputs
    has_outputs = bool(new_outcome.results) or covered_shots > 0
    if has_outputs:
        print("[resume] Collecting results...")
        metric_names = collect_results(chunk_dir, output_dir, batch_num, decoder_info.metric_name)
    else:
        metric_names = []

    if args.cleanup_intermediate:
        cleanup_intermediate(chunk_dir)

    incomplete_path = output_dir / INCOMPLETE_SHOTS_FILENAME
    if new_outcome.incomplete:
        write_incomplete_shots(incomplete_path, new_outcome.incomplete)
    elif incomplete_path.exists():
        incomplete_path.unlink()

    end_time = datetime.now(timezone.utc)

    metadata = build_decoding_metadata(
        decoder_info=decoder_info,
        dem_path=dem_path,
        dets_paths=(dets_path,),
        num_workers=num_workers,
        start_time=start_time,
        end_time=end_time,
        results=new_outcome.results,
        metrics_recorded=metric_names,
        incomplete_ranges=new_outcome.incomplete,
    )
    metadata_path = metadata_file_path(output_dir, batch_num)
    write_metadata(metadata_path, metadata)

    elapsed = (end_time - start_time).total_seconds()
    print(f"[resume] Done in {elapsed:.1f}s. Metadata → {metadata_path}")

    if new_outcome.incomplete:
        total = sum(r.shot_id_end - r.shot_id_start for r in new_outcome.incomplete)
        raise IncompleteTasksError(
            f"{len(new_outcome.incomplete)} task(s) covering {total} shot(s) did not "
            f"complete. Outputs for completed shots were written to {output_dir}. "
            f"See {incomplete_path} for affected shot_id ranges."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
