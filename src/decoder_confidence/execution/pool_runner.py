from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from decoder_confidence.execution.models import (
    IncompleteRange,
    RunOutcome,
    SimulationTask,
    WorkerConfig,
    WorkerResult,
)
from decoder_confidence.execution.worker import init_worker, run_task


def run_pool_tasks(
    tasks: list[SimulationTask],
    *,
    pool: Any,
    ctx: Any,
    worker_config: WorkerConfig,
    num_workers: int,
    maxtasksperchild: int | None,
    task_timeout_s: float,
    max_retries: int,
    poll_interval_s: float = 1.0,
    on_result: Callable[[WorkerResult], None] | None = None,
    abort_on_error: bool = False,
) -> RunOutcome:
    """Run ``tasks`` via ``pool.apply_async``, detecting hangs/timeouts.

    - The first cycle uses the supplied (already-spawned, caller-owned)
      ``pool``. Subsequent cycles spawn fresh pools via
      ``ctx.Pool(processes=num_workers, initializer=init_worker,
      initargs=(worker_config,), maxtasksperchild=maxtasksperchild)``.
    - If a task does not become ready within ``task_timeout_s`` it is
      considered hung ("the culprit"): the whole pool is terminated (a
      single ``multiprocessing.Pool`` cannot kill an individual worker), and
      the culprit is retried (consuming one of ``max_retries`` attempts) in a
      fresh pool. Any other task that was still in flight when the pool was
      terminated ("collateral") is re-submitted in the next cycle *without*
      consuming a retry attempt — it wasn't slow, it was just unlucky enough
      to share a pool generation with the culprit. This bounds the total
      number of pool-recreation cycles to at most
      ``len(tasks) * (max_retries + 1)``.
    - Tasks whose worker raised an exception (``status == "error"``) either
      abort the whole run immediately when ``abort_on_error`` is true, or are
      retried up to ``max_retries`` times (without needing a pool restart)
      before being recorded as an ``IncompleteRange(reason="error", ...)``.
    - Tasks that exhaust their retries (timeout or error) are converted to
      ``IncompleteRange`` entries and reported to ``on_result`` as a
      synthetic ``status="error"`` ``WorkerResult`` (with ``output_path =
      Path()``, which is never read for non-ok/skipped statuses) so callers'
      existing progress-counting/printing logic needs no special-casing.
    - When ``abort_on_error`` is false, never raises for partial failure.
      Returns once every task has either succeeded, been skipped, or been
      converted to an ``IncompleteRange``.
    """
    results: list[WorkerResult] = []
    incomplete: list[IncompleteRange] = []

    pending: list[tuple[SimulationTask, int]] = [(task, 1) for task in tasks]
    current_pool = pool
    owns_pool = False

    while pending:
        submitted: dict[int, tuple[SimulationTask, int, Any, float]] = {}
        for idx, (task, attempt) in enumerate(pending):
            async_result = current_pool.apply_async(run_task, (task,))
            submitted[idx] = (task, attempt, async_result, time.monotonic())

        next_pending: list[tuple[SimulationTask, int]] = []
        terminated = False

        while submitted:
            time.sleep(poll_interval_s)
            now = time.monotonic()
            culprit_idxs: list[int] = []

            for idx, (task, attempt, async_result, submit_time) in list(submitted.items()):
                if async_result.ready():
                    try:
                        result = async_result.get(timeout=0)
                    except Exception as exc:
                        result = WorkerResult(
                            status="error",
                            duration_s=now - submit_time,
                            output_path=Path(),
                            num_shots=task.num_shots,
                            batch_id=task.batch_id,
                            message=str(exc),
                        )

                    if result.status in {"ok", "skipped"}:
                        results.append(result)
                        if on_result is not None:
                            on_result(result)
                    else:  # "error"
                        if abort_on_error:
                            current_pool.terminate()
                            current_pool.join()
                            raise RuntimeError(
                                f"Worker task failed for batch_id={task.batch_id}: "
                                f"{result.message}"
                            )
                        if attempt <= max_retries:
                            next_pending.append((task, attempt + 1))
                        else:
                            incomplete.append(
                                IncompleteRange.from_task(
                                    task,
                                    reason="error",
                                    message=result.message,
                                    attempts=attempt,
                                )
                            )
                            if on_result is not None:
                                on_result(
                                    WorkerResult(
                                        status="error",
                                        duration_s=result.duration_s,
                                        output_path=Path(),
                                        num_shots=task.num_shots,
                                        batch_id=task.batch_id,
                                        message=(
                                            f"failed after {attempt} attempt(s): "
                                            f"{result.message}"
                                        ),
                                    )
                                )
                    del submitted[idx]
                elif now - submit_time > task_timeout_s:
                    culprit_idxs.append(idx)

            if culprit_idxs:
                current_pool.terminate()
                current_pool.join()
                terminated = True

                culprit_set = set(culprit_idxs)
                for idx, (task, attempt, _async_result, _submit_time) in submitted.items():
                    if idx in culprit_set:
                        if attempt <= max_retries:
                            next_pending.append((task, attempt + 1))
                        else:
                            incomplete.append(
                                IncompleteRange.from_task(
                                    task,
                                    reason="timeout",
                                    message=(
                                        f"exceeded {task_timeout_s:.1f}s "
                                        f"(attempt {attempt})"
                                    ),
                                    attempts=attempt,
                                )
                            )
                            if on_result is not None:
                                on_result(
                                    WorkerResult(
                                        status="error",
                                        duration_s=task_timeout_s,
                                        output_path=Path(),
                                        num_shots=task.num_shots,
                                        batch_id=task.batch_id,
                                        message=(
                                            f"timed out after {attempt} attempt(s) "
                                            f"(>{task_timeout_s:.1f}s)"
                                        ),
                                    )
                                )
                    else:
                        # Collateral: this pool generation was torn down
                        # because of a sibling's timeout, not this task's
                        # own slowness. Re-run without spending a retry.
                        next_pending.append((task, attempt))

                submitted.clear()
                break

        if terminated and next_pending:
            current_pool = ctx.Pool(
                processes=num_workers,
                initializer=init_worker,
                initargs=(worker_config,),
                maxtasksperchild=maxtasksperchild,
            )
            owns_pool = True

        pending = next_pending

    if owns_pool:
        current_pool.close()
        current_pool.join()

    return RunOutcome(results=results, incomplete=incomplete)
