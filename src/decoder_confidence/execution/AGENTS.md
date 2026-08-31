# AGENTS — `execution/`

Worker-pool orchestration shared by `decoding/__main__.py` and
`decoding/resume.py`. See the root [AGENTS.md](../../../AGENTS.md) and `ARCHITECTURE.md`'s "The
execution model" first.

## Modules

| Module | Responsibility |
|---|---|
| `models.py` | `DecoderBase`, `DecoderFactory`, `SharedEnvDecoderFactory`, `SimulationTask`/`WorkerConfig`/`WorkerResult`/`ExecutionConfig`/`RunOutcome`/`IncompleteRange`/`IncompleteTasksError` |
| `manager.py` | `run_manager` — routes to threaded or multiprocessing executor; multiprocessing orchestration |
| `pool_runner.py` | `run_pool_tasks` — `apply_async` submission, hang/timeout detection, retry, collateral resubmit |
| `threaded_manager.py` | `ThreadPoolExecutor`-based manager for `SharedEnvDecoderFactory` decoders |
| `worker.py` | `init_worker`/`run_task` — per-process decoder construction and task execution |
| `batching.py` | `estimate_shots_per_task`, `estimate_task_timeout_s` |
| `hashing.py` | `stable_task_id`, `chunk_filename` — deterministic per-task chunk paths |
| `_execution_utils.py` | File discovery, core-affinity selection, part merging, RSS reporting |

## Contract

- Routing is `isinstance(config.decoder_factory, SharedEnvDecoderFactory)`
  — currently true only for Gurobi-backed ILP. Adding a new
  `SharedEnvDecoderFactory` implementer automatically routes it to the
  thread-pool path; anything else automatically goes through the
  multiprocessing path. Don't special-case a decoder name here — implement
  the right factory protocol instead.
- `init_worker` builds the decoder **once per worker process** (pool
  `initializer`), not once per task — `run_task` must keep reusing
  `_STATE.decoder` rather than rebuilding it, or the whole point of
  `maxtasksperchild` (process recycling interval, not per-task) is lost.
- `stable_task_id`/`chunk_filename` must stay a pure function of
  `(decoder_name, dets_path, start_shot_index, num_shots, batch_id,
  shot_id_offset)` — `run_pool_tasks`'s retry/collateral-resubmit logic
  depends on a resubmitted task mapping to the same chunk path.
- Any new `ctx.Pool(...)` construction must keep `mp.get_context("spawn")`
  — required because Gurobi/CPLEX hold C-level license/solver state that
  does not survive `fork()` safely.

## Pitfalls

- The initial probe in `_run_manager_multiprocess` (`manager.py:136`) and
  `resume.py`'s non-threaded probe (`resume.py:570`) run via a
  **synchronous** `pool.apply(...)`, with no timeout protection —
  `FUTURE.md` §1.2 (GitHub issue #2, open). Don't assume every task path is
  timeout-protected; this one specifically isn't.
- `ExecutionConfig.timeout_multiplier` defaults to `100000.0` and isn't
  exposed as a CLI flag on the main `decoding` entry point — in practice an
  unbounded per-task timeout there. `resume.py` exposes a sane `10.0`
  default. `FUTURE.md` §1.3.
- `_ilp_cplex_logicalgap.py`'s factory is a plain callable specifically so
  it takes the multiprocessing path (CPLEX has no session limit, unlike
  Gurobi) — don't "fix" it into a `SharedEnvDecoderFactory` without checking
  whether that assumption still holds for the CPLEX license in use.

## Tests

`test_pool_runner.py` (fake `Pool`/`AsyncResult` objects — retry/timeout/
collateral logic tested deterministically, no real multiprocessing),
`test_batching.py`, `test_worker_init_failure.py` (worker-initializer
exception handling). No test exercises the real Gurobi/CPLEX solver paths
outside the e2e tests in `tests/` (see `decoding/AGENTS.md`).
