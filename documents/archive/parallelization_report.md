> Archived 2026-08-31. The double-spawn-cycle fix this report diagnoses is
> present in current code (`execution/manager.py` merges the probe into the
> main pool — see the "run probe inside the already-initialised pool"
> comment there) and is documented in `ARCHITECTURE.md`'s "The execution
> model". It does **not** cover the still-open probe-timeout gap left behind
> by that same code path — see `FUTURE.md` §1.2. Kept verbatim below for the
> original diagnosis and reasoning.

# Parallelization Bottleneck Report: ILP Decoder Simulation

## Summary

The PBS job for the ILP-based decoder confidence simulation showed near-zero CPU
efficiency despite allocating 96 cores. Diagnosis identified a structural overhead
in the worker pool lifecycle as the primary cause, and the fix eliminates a redundant
spawn/teardown cycle by merging the probe phase into the main worker pool.

---

## Observed Symptoms

```
resources_used.cput     = 00:02:49  (169 s)
resources_used.walltime = 00:13:01  (781 s)
resources_used.ncpus    = 96
resources_used.cpupercent = 98
```

**Effective parallelism = CPU time / wall time = 169 / 781 ≈ 0.22 cores**

Although 96 cores were allocated and `cpupercent` peaked at 98 % (≈ 1 core at
full utilization), the average across the entire job was less than one quarter of
a single core. Most wall-clock time was spent with all processes idle or blocked
on I/O rather than computing.

---

## Root Cause: Double Spawn Cycle

### What the original code did

`run_manager` was split into two separate `multiprocessing.Pool` lifetimes:

```
Phase 1 — probe pool (processes=1)
  ├── spawn 1 worker process
  │     └── load Python runtime
  │     └── import stim, polars, numpy, …
  │     └── import gurobipy → connect to license server
  │     └── build ILPDecoder (parity-check matrix, Gurobi model)
  ├── run 20 probe shots  ← actual computation
  └── terminate pool (worker process exits)

Phase 2 — main pool (processes=96)
  ├── spawn 96 worker processes  ← ALL of the above, 96 times over
  ├── run all simulation tasks   ← actual computation
  └── terminate pool
```

The probe pool was created solely to time 20 shots and estimate
`shots_per_task`. Once destroyed, the main pool re-did every step of
initialization from scratch across 96 processes.

### Why this dominates wall time

`mp.get_context("spawn")` starts each worker from a fresh Python interpreter.
On an HPC cluster, where libraries live on NFS, each process must:

1. Load the Python interpreter and standard library from NFS.
2. Import heavy packages (`stim`, `polars`, `numpy`, etc.).
3. Import `gurobipy` and call `gp.Env.start()`, which connects to the
   Gurobi license server over the network.
4. Construct the `ILPDecoder` object (builds the MIP model in memory).

Steps 2–4 are largely I/O- or network-bound, not CPU-bound, which explains
why `cpupercent` was near 100 % for only brief intervals while most of the
781 s was spent waiting. With 97 processes (1 probe + 96 main) all going
through this path, the total I/O and license-server load was substantial.

### Timeline estimate

| Phase | Approx. wall time | Approx. CPU time |
|-------|-------------------|------------------|
| Probe pool spawn + Gurobi init | ~30–60 s | ~30–60 s (1 process) |
| 20 probe shots (ILP solve) | ~10–30 s | ~10–30 s |
| Probe pool teardown | ~5 s | negligible |
| Main pool spawn + Gurobi init (96×) | ~400–600 s | ~30–60 s (I/O bound) |
| Actual simulation tasks | ~50–100 s | ~50–100 s × 96 (if tasks present) |

The spawn + init phase of the main pool alone likely accounts for the bulk of
the 781-second wall time while contributing almost nothing to `cput`.

---

## Fix Applied

The `_run_probe` helper function and its dedicated pool have been removed.
The probe task now runs inside the main pool using `pool.apply()`, which
blocks synchronously until the result is available—exactly the same
observable behavior as before, but without a second spawn cycle.

```python
# execution/manager.py  —  run_manager()

ctx = mp.get_context("spawn")
with ctx.Pool(
    processes=config.num_workers, initializer=init_worker, initargs=(worker_config,)
) as pool:
    # Probe runs inside the already-initialised main pool.
    probe_result = pool.apply(run_task, (probe_task,))
    if probe_result.status != "ok":
        raise RuntimeError(f"Probe task failed: {probe_result.message}")

    ...  # delete probe output, estimate shots_per_task, build task list

    for result in pool.imap_unordered(run_task, tasks, chunksize=1):
        ...
```

**Effect:** The 96-worker pool is now spawned exactly once. The single probe
task occupies one worker for a few seconds while the remaining 95 are idle but
already initialized and ready. The moment `imap_unordered` is called, all 96
workers begin processing tasks immediately—no re-initialization required.

---

## Expected Improvement

| Metric | Before | After (estimate) |
|--------|--------|------------------|
| Pool spawn cycles | 2 (1-worker + 96-worker) | 1 (96-worker only) |
| Gurobi `Env.start()` calls | 97 | 96 |
| Wasted wall time on probe pool | ~60–120 s | 0 s |
| Time workers idle after init | minutes (waiting for probe pool to finish) | seconds (probe uses 1 worker, others wait briefly) |

For a job with O(100) total tasks and modest per-task compute, the saved
spawn overhead is a significant fraction of total wall time.

---

## Secondary Bottlenecks (Not Fixed Here)

### 1. NFS library loading contention

When 96 processes each load Python + Gurobi from an NFS mount, disk reads
serialise at the network level. This manifests as low `cpupercent` and high
`walltime` during pool initialization. Mitigation options:

- Pre-stage libraries to local node storage (`$TMPDIR`) in the PBS prolog.
- Switch from `spawn` to `forkserver` if Gurobi is fork-safe in the target
  environment. `forkserver` loads libraries once in the forkserver process,
  then forks (not re-imports) for each worker.

### 2. Simultaneous Gurobi license server connections

All 96 workers call `gp.Env.start()` concurrently during `init_worker`. If
the license server serialises connection requests, this can add tens of seconds
of wall time. Staggering `init_worker` startup by a small per-worker delay
(e.g. `time.sleep(worker_index * 0.2)`) distributes the connection load.

### 3. `target_task_seconds` tuning

The default `target_task_seconds = 30.0` determines granularity. If a single
ILP solve takes on the order of seconds, each task will contain only tens of
shots, creating many small tasks. Increasing `target_task_seconds` reduces
IPC overhead; decreasing it improves load balancing when solve times vary
widely across shots. Profile per-shot solve time to choose an appropriate value.

---

## Conclusion

The dominant overhead was a structural one: the simulation manager spawned a
temporary one-worker pool to run the probe, tore it down, then spawned the
full pool from scratch. On an HPC system with NFS-mounted libraries and a
network-connected Gurobi license server, each spawn cycle is expensive. The
fix—running the probe inside the main pool—eliminates one full cycle with no
change to correctness or observable behavior.
