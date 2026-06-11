"""Unit tests for execution.pool_runner.run_pool_tasks.

Uses fake Pool/AsyncResult/ctx objects (no real multiprocessing) so the
retry/timeout/collateral-requeue logic can be tested deterministically and
quickly. Real multiprocess coverage (init_worker/run_task via ctx.Pool) comes
from the e2e tests in test_e2e_pipeline.py and test_obs_flip_idx_e2e.py.
"""
from __future__ import annotations

import time
from pathlib import Path

from decoder_confidence.execution.models import (
    SimulationTask,
    WorkerConfig,
    WorkerResult,
)
from decoder_confidence.execution.pool_runner import run_pool_tasks


def _make_task(*, batch_id: int, num_shots: int = 4) -> SimulationTask:
    return SimulationTask(
        dets_path=Path(f"/data/det_batch={batch_id}.b8"),
        start_shot_index=0,
        num_shots=num_shots,
        batch_id=batch_id,
        shot_id_offset=batch_id * 1000,
    )


def _ok_result(task: SimulationTask) -> WorkerResult:
    return WorkerResult(
        status="ok",
        duration_s=0.001,
        output_path=Path(f"/tmp/chunk_{task.batch_id}.parquet"),
        num_shots=task.num_shots,
        batch_id=task.batch_id,
    )


def _error_result(task: SimulationTask, message: str) -> WorkerResult:
    return WorkerResult(
        status="error",
        duration_s=0.001,
        output_path=Path(),
        num_shots=task.num_shots,
        batch_id=task.batch_id,
        message=message,
    )


def _dummy_worker_config() -> WorkerConfig:
    return WorkerConfig(
        dem_path=Path("/dev/null"),
        output_dir=Path("/tmp"),
        decoder_factory=lambda: None,
    )


class _FakeAsyncResult:
    """Stand-in for multiprocessing.pool.AsyncResult.

    ``ready_after=None`` means "never becomes ready" (simulates a hung
    worker); otherwise it becomes ready ``ready_after`` seconds after
    construction.
    """

    def __init__(self, *, ready_after: float | None, value: WorkerResult | None = None):
        self._ready_at = None if ready_after is None else time.monotonic() + ready_after
        self._value = value

    def ready(self) -> bool:
        if self._ready_at is None:
            return False
        return time.monotonic() >= self._ready_at

    def get(self, timeout: float = 0) -> WorkerResult:
        assert self._value is not None
        return self._value


class _FakePool:
    """Stand-in for multiprocessing.pool.Pool: scripted apply_async results."""

    def __init__(self, script):
        self._script = script
        self.terminate_count = 0

    def apply_async(self, func, args):
        (task,) = args
        return self._script(task)

    def terminate(self) -> None:
        self.terminate_count += 1

    def join(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeCtx:
    """Stand-in for mp.get_context("spawn"): spawns more _FakePool instances."""

    def __init__(self, script):
        self._script = script
        self.pools_created: list[_FakePool] = []

    def Pool(self, **kwargs):
        pool = _FakePool(self._script)
        self.pools_created.append(pool)
        return pool


def test_run_pool_tasks_all_succeed_first_cycle():
    tasks = [_make_task(batch_id=i) for i in range(3)]

    def script(task: SimulationTask) -> _FakeAsyncResult:
        return _FakeAsyncResult(ready_after=0.0, value=_ok_result(task))

    pool = _FakePool(script)
    ctx = _FakeCtx(script)

    outcome = run_pool_tasks(
        tasks,
        pool=pool,
        ctx=ctx,
        worker_config=_dummy_worker_config(),
        num_workers=2,
        maxtasksperchild=None,
        task_timeout_s=10.0,
        max_retries=1,
        poll_interval_s=0.01,
    )

    assert outcome.incomplete == []
    assert {r.batch_id for r in outcome.results} == {0, 1, 2}
    assert all(r.status == "ok" for r in outcome.results)
    assert pool.terminate_count == 0
    assert ctx.pools_created == []


def test_run_pool_tasks_error_exhausts_retries():
    ok_task = _make_task(batch_id=0)
    err_task = _make_task(batch_id=1)
    tasks = [ok_task, err_task]

    seen_results: list[WorkerResult] = []

    def script(task: SimulationTask) -> _FakeAsyncResult:
        if task.batch_id == err_task.batch_id:
            return _FakeAsyncResult(ready_after=0.0, value=_error_result(task, "boom"))
        return _FakeAsyncResult(ready_after=0.0, value=_ok_result(task))

    pool = _FakePool(script)
    ctx = _FakeCtx(script)

    outcome = run_pool_tasks(
        tasks,
        pool=pool,
        ctx=ctx,
        worker_config=_dummy_worker_config(),
        num_workers=2,
        maxtasksperchild=None,
        task_timeout_s=10.0,
        max_retries=1,
        poll_interval_s=0.01,
        on_result=seen_results.append,
    )

    # The good task lands in results; the always-failing task is retried
    # max_retries+1 = 2 times total, then recorded as incomplete.
    assert {r.batch_id for r in outcome.results} == {ok_task.batch_id}
    assert all(r.status == "ok" for r in outcome.results)

    assert len(outcome.incomplete) == 1
    incomplete = outcome.incomplete[0]
    assert incomplete.batch_id == err_task.batch_id
    assert incomplete.reason == "error"
    assert incomplete.attempts == 2
    assert "boom" in (incomplete.message or "")

    # Error retries don't require a fresh pool (no termination needed).
    assert pool.terminate_count == 0
    assert ctx.pools_created == []

    # on_result is called for the ok task and for the synthetic
    # failure WorkerResult of the exhausted task.
    statuses_by_batch = {r.batch_id: r.status for r in seen_results}
    assert statuses_by_batch[ok_task.batch_id] == "ok"
    assert statuses_by_batch[err_task.batch_id] == "error"


def test_run_pool_tasks_timeout_with_collateral_requeue():
    """A hung task triggers pool termination; an in-flight sibling that has
    not itself timed out is "collateral" -- requeued without consuming a
    retry -- and everything eventually resolves successfully.
    """
    TASK_TIMEOUT_S = 0.15
    POLL_INTERVAL_S = 0.05
    STAGGER_S = 0.2

    ok_task = _make_task(batch_id=0)
    hang_task = _make_task(batch_id=1)
    collateral_task = _make_task(batch_id=2)
    tasks = [ok_task, hang_task, collateral_task]

    call_counts: dict[int, int] = {}

    def script(task: SimulationTask) -> _FakeAsyncResult:
        call_counts[task.batch_id] = call_counts.get(task.batch_id, 0) + 1
        n = call_counts[task.batch_id]

        if task.batch_id == ok_task.batch_id:
            return _FakeAsyncResult(ready_after=0.0, value=_ok_result(task))

        if task.batch_id == hang_task.batch_id:
            if n == 1:
                # Never becomes ready -- simulates a hung/OOM-killed worker.
                return _FakeAsyncResult(ready_after=None)
            return _FakeAsyncResult(ready_after=0.0, value=_ok_result(task))

        # collateral_task: on its first submission, stagger the recorded
        # submit_time (by sleeping inside apply_async) so that, at the poll
        # tick where hang_task is flagged as the timeout culprit, this task
        # has *not yet* exceeded task_timeout_s itself.
        if n == 1:
            time.sleep(STAGGER_S)
            return _FakeAsyncResult(ready_after=10.0, value=_ok_result(task))
        return _FakeAsyncResult(ready_after=0.0, value=_ok_result(task))

    pool = _FakePool(script)
    ctx = _FakeCtx(script)

    outcome = run_pool_tasks(
        tasks,
        pool=pool,
        ctx=ctx,
        worker_config=_dummy_worker_config(),
        num_workers=2,
        maxtasksperchild=None,
        task_timeout_s=TASK_TIMEOUT_S,
        max_retries=1,
        poll_interval_s=POLL_INTERVAL_S,
    )

    assert outcome.incomplete == []
    assert {r.batch_id for r in outcome.results} == {0, 1, 2}
    assert all(r.status == "ok" for r in outcome.results)

    # Exactly one pool termination (for the hung task) and one fresh pool
    # spawned to retry it + the collateral task.
    assert pool.terminate_count == 1
    assert len(ctx.pools_created) == 1

    # hang_task needed 2 submissions (timeout, then succeeded on retry);
    # collateral_task needed 2 submissions (collateral requeue, then
    # succeeded) without ever being a "culprit" itself.
    assert call_counts[hang_task.batch_id] == 2
    assert call_counts[collateral_task.batch_id] == 2
