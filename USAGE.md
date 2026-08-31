# USAGE

Operational manual for the decoder-confidence simulation pipeline. See
[README.md](README.md) for a project overview and quickstart, [ARCHITECTURE.md](ARCHITECTURE.md) for
how the code is put together, and [FUTURE.md](FUTURE.md) for known bugs and gaps referenced
below.

## Contents

1. [Prerequisites](#prerequisites)
2. [Stage 2: Sampling](#stage-2-sampling)
3. [Stage 3: Decoding](#stage-3-decoding)
4. [Configuration files](#configuration-files)
5. [Parallel execution](#parallel-execution)
6. [Resume and incomplete shots](#resume-and-incomplete-shots)
7. [Cluster execution](#cluster-execution)
8. [Analysis](#analysis)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python `>=3.10`; the project has been exercised under the `decoder_confidence`
  conda environment (`environment.yml` is a snapshot of it, see [README.md#installation](README.md#installation)).
- `PYTHONPATH` must include `src/` to run `decoder_confidence.*` modules, and the
  **repository root** to run anything under `analysis/` (it is not an installable
  package). `commands/*.sh` scripts set `PYTHONPATH="$ROOT_DIR/src"` themselves.
- Optional dependencies, one group per decoder (`pip install -e ".[<extra>]"`):

  | Decoder | Extra | Package(s) |
  |---|---|---|
  | `BP-LSD`, `VIBE-LSD` | `bplsd` | `ldpc` |
  | `MWPM` | `mwpm` | `pymatching` |
  | `ILP` (Gurobi) | `ilp` | `gurobipy`, `ilp-decoder` |
  | `ILP` (CPLEX) | *(none)* | `cplex`, `docplex` — not declared in `pyproject.toml`/`environment.yml`; install separately (see [FUTURE.md](FUTURE.md)) |
  | `RELAY-BP` | `relay` | `relay-bp` (built from source via `maturin`; not on PyPI) |

  Each is imported lazily and guarded — a missing package raises a clear
  `RuntimeError` (or, for `ilp_decoder`, an `ImportError` from an unguarded
  module-level import reached only when that solver is actually selected) rather
  than failing at `import decoder_confidence`.

## Stage 2: Sampling

**[documents/STEP2_SAMPLING.md](documents/STEP2_SAMPLING.md) is the authoritative reference** for this stage —
internal pipeline, `--sampling_method` semantics, and idempotence. This section
is a compact index.

```
python -m decoder_confidence.sampling [flags]
```

| Flag | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `--code` | str | yes | — | Code family, matched against `circuits/*.stim` filenames |
| `--out_dir` | str | yes | — | Root of the output layout (below) |
| `--noise_model` | str | yes | — | e.g. `uniform`, `si1000`, `bitflip`, `phenomenological` |
| `--rounds` | str | yes | — | Kept as a string; some circuits encode rounds symbolically |
| `--d` | int | yes | — | Code distance, `>= 1` |
| `--p` | float | yes | — | Physical error rate, matched to the circuit filename with `abs_tol=1e-12` |
| `--num_shots` | int | yes | — | Total shots across all batches |
| `--det_sample_seed` | int | yes | — | Base seed, must be in `[0, 2**64 - 1]` |
| `--num_batch` | int | yes | — | `1 <= num_batch <= num_shots` |
| `--xyz_decoding` | bool | no | `False` | Keep both X/Z detector bases if `True` |
| `--sampling_method` | `unified`\|`per_batch_seed` | no | `unified` | See STEP2_SAMPLING.md — **use `unified` for all new data** |

**`--sampling_method`, in one paragraph:** `unified` draws all `num_shots` in a
single call keyed on `det_sample_seed` and slices deterministically into
batches, so the sampled shot set never depends on `num_batch`. `per_batch_seed`
(legacy, pre-2026-06-25) compiles one sampler per batch with
`seed = det_sample_seed + (batch_index - 1)`, so the sampled shots **do**
depend on `num_batch` — two runs with the same seed but different `num_batch`
are not reproductions of each other. Only use `per_batch_seed` to reproduce or
compare against data generated before the fix; `metadata.json` records which
method was actually used.

**Idempotence:** if `dem.dem` or a given `det_batch=N.b8` already exists, it is
left untouched (a log line is emitted instead of overwriting it) — re-running
the same command is safe and only fills in missing files.

Output layout: `<out_dir>/<circuit_stem>,xyz=<bool>/{dem.dem, metadata.json,
sampled_data/det_batch=N.b8}` — full detail in STEP2_SAMPLING.md.

## Stage 3: Decoding

```
python -m decoder_confidence.decoding [flags]
```

| Flag | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `--code` | str | yes | — | Matched against a circuit directory name (see below) |
| `--data_dir` | str | yes | — | Root under which circuit directories are searched (`rglob`) |
| `--noise_model` | str | yes | — | |
| `--rounds` | str | yes | — | |
| `--d` | int | yes | — | |
| `--p` | float | yes | — | Matched with `abs_tol=1e-12` |
| `--batch_num` | int | yes | — | `>= 1`; selects `sampled_data/det_batch=<N>.b8` |
| `--num_workers` | int | yes | — | Processes (most decoders) or solver threads (`SharedEnvDecoderFactory`, e.g. Gurobi ILP) — see [Parallel execution](#parallel-execution) |
| `--decoder_config` | str | yes | — | Path to a `decoder`/`metric`/`decoder_options`/`metric_options` YAML file, see [Configuration files](#configuration-files) |
| `--xyz` | bool | yes | — | Must match the `xyz=` token of the target directory |
| `--ibm_reproduce` | flag | no | `False` | If set, require `ibm_reproduce=True` in the directory name; if unset, require it absent |
| `--verbose` | bool | no | `False` | Print progress/RSS during decoding |
| `--max_chunk_files` | int | no | `1000` | Merge chunk files into parts once this count is exceeded (`<=0` disables) |
| `--merge_chunk_group_size` | int | no | `None` → `min(200, max_chunk_files)` | Chunks merged per part file |
| `--cleanup_intermediate` | bool | no | `True` | Remove chunk/part parquets after final outputs are written |

**Circuit-directory lookup:** `_find_circuit_dir` (`decoding/__main__.py`)
`rglob`s `--data_dir` for directories whose name parses as
`code=,d=,rounds=,noisemodel=,p=[,xyz=][,ibm_reproduce=True]` and matches every
supplied flag (`--code`, `--d`, `--rounds`, `--noise_model`, `--p`, `--xyz`,
`--ibm_reproduce`). `rounds=` also accepts a legacy `r=` key.

- **Zero matches** → `FileNotFoundError`, listing the expected directory-name
  pattern it was looking for.
- **More than one match** → `FileExistsError`, listing every matched path. This
  can happen if, e.g., two sampling runs used the same parameters into
  different `--out_dir`s that both sit under `--data_dir`.

For `decoder: ILP`, `decoder_options.threads * --num_workers` is checked
against `len(os.sched_getaffinity(0))` (falling back to `os.cpu_count()`); it
raises `ValueError` if exceeded, regardless of which ILP solver is configured.

## Configuration files

### Schema

```yaml
decoder: <name>            # required; case/hyphen-insensitive, see README.md#supported-decoders
metric: <name>              # required; see README.md#supported-confidence-metrics
decoder_options: {}          # optional mapping, decoder-specific (below)
metric_options: {}           # optional mapping, metric-specific (below)
```

Dispatch order and exact accepted `(decoder, metric)` pairs are documented in
[ARCHITECTURE.md](ARCHITECTURE.md#the-decoder-abstraction) — order matters, since some
branches shadow others.

### `decoder_options` by decoder

| Decoder | Key | Type | Default | Effect |
|---|---|---|---|---|
| `ILP` | `solver` | str | **required** | `gurobi` or `cplex`; selects the backend module |
| | `threads` | int | `1` | Passed to the solver; also used in the `threads × num_workers` core check |
| | `time_limit_s` | float | solver default | Per-shot solve time limit |
| | `mip_gap` | float | solver default | MIP optimality gap |
| | `log_to_console` | bool | `False` | Solver console output |
| | `enable_lazy_constraints` | bool | `False` | |
| | `random_seed` | int | solver default | |
| | `solver_options` | mapping | `{}` | Gurobi only, forwarded to the solver |
| `BP-LSD` | *(forwarded)* | — | — | Passed as `**kwargs` to `ldpc.bplsd_decoder.BpLsdDecoder` after defaulting `input_vector_type="syndrome"` — full key set is `(unverified: see the ldpc library)`. Observed in `conf/*.yaml`: `max_iter`, `bp_method`, `lsd_order` |
| `MWPM` | *(forwarded)* | — | — | Passed to `pymatching.Matching.from_check_matrix`; `weights`/`spacelike_weights`/`error_probabilities`/`faults_matrix` are rejected (managed internally) |
| `RELAY-BP` | *(forwarded)* | — | — | Passed to `relay_bp.bp.RelayDecoderF64`, filtered against its live constructor signature — unknown keys raise `ValueError` listing what's supported. Observed in `conf/*.yaml`: `gamma0`, `gamma_dist_interval`, `pre_iter`, `set_max_iter`, `num_sets`, `stop_nconv`, `seed` |
| `VIBE-LSD` | `ensemble_size` | int | `32` | Number of BP instances (distinct serial schedules) |
| | `correction_limit` | int | `5` | Stop collecting candidates after this many converge |
| | `max_iter` | int | `25` | BP iterations per instance |
| | `bp_method` | str | `"minimum_sum"` | |
| | `ms_scaling_factor` | float | `0.625` | |
| | `lsd_order` | int | `0` | LSD fallback order |
| | `prior_perturbation` | float | `0.0` | Per-instance prior perturbation width δ |
| | `seed` | int | `0` | |

### `metric_options` by metric

| Metric | Key | Type | Default | Effect |
|---|---|---|---|---|
| `forced_gap_ml` | `get_all_failure_rate` | bool | `False` | Emit `forced_gap_ml_case` labels |
| | `get_detail_stat` | bool | `False` | Emit per-stage detail-stat columns |
| | `random_split` | bool | `False` | Use random-split constrained decoding |
| | `n_splits` | int | `3` | |
| | `split_seed` | int | `0` | |
| | `split_balanced` | bool | `False` | |
| | `alpha` / `cluster_llr_alpha` | float | `2.0` | Alpha-norm order when `cluster_llr` is computed as a detail stat |
| | `forced_unconverged_confidence_value` | `"positive"`\|`"negative"` | `None` | RELAY-BP non-convergence handling — see `documents/relay_bp_nonconvergence_behavior.md` |
| `linearize_logicalgap` | *(same as `forced_gap_ml` minus `get_all_failure_rate`)* | | | |
| `reweighted_linearized_gap` | `b` | float | **required** | Ratio-reweighting exponent, must be `> 1.0` |
| | `forced_unconverged_confidence_value` | `"positive"`\|`"negative"` | `None` | |
| `ar-pec` / `ar-lec` | `test_type` | `"Ratio"`\|`"Gap"` | **required** | |
| | `b` | float | **required** | `> 1.0` for `Ratio`, `> 0.0` for `Gap` |
| | `num_decoding_rounds` | int | **required** | `>= 2` |
| `wills_reproduce` | `forced_num_sets` | int | **required** | Overrides `decoder_options.num_sets` for the K forced runs only (paper Appendix A) |
| | `get_detail_stat` | bool | `False` | |
| `cluster_llr` | `alpha` | float or `"inf"` | `2.0` | Alpha-norm order; `"inf"` selects the max-cluster variant |
| `forcing_degradation_test` | `alpha` (or legacy `b`) | float | `2.0` | |

### `conf/*.yaml` — what each file is

| File | `decoder` | `metric` | Notes |
|---|---|---|---|
| `step3_config.yaml` | `ILP` | `logical_gap` | `solver: cplex`, `threads: 1`; used by the README quickstart and `test_step3_decoding_e2e` |
| `step3_config_cplex.yaml` | `ILP` | `logical_gap` | Byte-identical content to `step3_config.yaml` (verified — only a trailing newline differs); used by `test_step3_decoding_cplex_e2e`. See [FUTURE.md](FUTURE.md) |
| `step3_config_ar_bplsd.yaml` | `BP-LSD` | `ar-lec` | `test_type: Ratio`, `b: 1.1`, `num_decoding_rounds: 2` |
| `step3_config_linearize_bplsd.yaml` | `BP-LSD` | **`cluster_llr`** | Filename says "linearize" but the config's `metric:` is `cluster_llr`, not `linearize_logicalgap` — a naming/content mismatch, see [FUTURE.md](FUTURE.md) |
| `step3_config_rlg_bplsd.yaml` | `BP-LSD` | `reweighted_linearized_gap` | `b: 2.0` |
| `step3_config_vibelsd_ar.yaml` | `VIBE-LSD` | `ar-lec` | |
| `step3_config_vibelsd_cluster_llr.yaml` | `VIBE-LSD` | `cluster_llr` | |
| `step3_config_vibelsd_linearize.yaml` | `VIBE-LSD` | `linearize_logicalgap` | |
| `step3_config_wills_reproduce.yaml` | `relay-bp` | `wills_reproduce` | Fig. 2(a) reproduction of arXiv:2605.20346; extensively self-documented in-file |
| `forcing_degradation_bplsd.yaml` | `BP-LSD` | `forcing_degradation_test` | |
| `forcing_degradation_relay_bp.yaml` | `RELAY-BP` | `forcing_degradation_test` | |

## Parallel execution

There are two independent levels of parallelism.

**Batch-level** — coarse, external to a single invocation: separate processes
or cluster jobs each run `python -m decoder_confidence.decoding ... --batch_num
N` for a different `N`, writing into the same `decoding_result/decoder=...,
metric=.../` directory concurrently. This is made safe by atomic output writes
(below), not by any coordination between the invocations.

**Worker-level** — inside one invocation, controlled by `--num_workers`.
Which mechanism is used depends on the decoder factory, decided in
`execution/manager.py:run_manager` by `isinstance(decoder_factory,
SharedEnvDecoderFactory)`:

- **`SharedEnvDecoderFactory`** (currently only Gurobi-backed `ILP` —
  `_ilp_logicalgap.py:_ILPDecoderFactory`) → `execution/threaded_manager.py`: a
  single shared Gurobi environment (one WLS session) and a `ThreadPoolExecutor`
  with `--num_workers` threads, so the C-layer solves run concurrently while
  only one license session is consumed.
- **Everything else, including CPLEX-backed `ILP`** (`_ilp_cplex_logicalgap.py`
  explicitly returns a plain callable — "CPLEX with an academic licence has no
  session limits" — routing it to the multiprocessing path) →
  `execution/manager.py:_run_manager_multiprocess`: a `multiprocessing.Pool`
  (`spawn` context) of `--num_workers` processes, run through
  `execution/pool_runner.py:run_pool_tasks`.

### Task granularity (multiprocessing path)

1. An initial probe of `min(ExecutionConfig.initial_probe_shots, shots in the
   first file)` shots (default `20`) is run **synchronously inside the main
   pool** (`pool.apply`, not `apply_async`) before any timeout protection
   exists — this specific call is exactly where GitHub issue #2's still-open
   probe-hang gap lives; see [FUTURE.md](FUTURE.md).
2. `shots_per_task = estimate_shots_per_task(probe_shots, probe_duration_s,
   target_task_seconds)` ≈ `(probe_shots / probe_duration_s) *
   target_task_seconds`, clamped to `[1, max_shots_per_task]`.
   `target_task_seconds` defaults to `30.0` and, like `max_shots_per_task`, is
   **not exposed as a CLI flag** on `python -m decoder_confidence.decoding` —
   only `resume.py` exposes the timeout-related knobs (below).
3. `task_timeout_s = max(timeout_multiplier * expected_task_s,
   min_task_timeout_s)`, where `expected_task_s = (probe_duration_s /
   probe_shots) * shots_per_task`. On the main `decoding` CLI,
   `timeout_multiplier` defaults to **`100000.0`** and `min_task_timeout_s` to
   `60.0` (`ExecutionConfig` defaults, unreachable from the CLI) — in practice
   this makes the post-probe per-task timeout essentially unbounded.
   `resume.py` exposes `--timeout_multiplier` (default `10.0`, far more
   conservative) and `--min_task_timeout_s` (default `60.0`).
4. Once running, `run_pool_tasks` submits every task via `apply_async` and
   polls every `poll_interval_s` (`1.0`s). A task exceeding `task_timeout_s` is
   the "culprit": the whole pool is terminated (a `Pool` cannot kill one
   worker), the culprit is retried (consuming one of `max_task_retries`
   attempts, default `1`), and any other in-flight "collateral" task is
   resubmitted in a fresh pool **without** consuming a retry. A task that
   errors (`status="error"`) either aborts the whole run immediately
   (`decoding/__main__.py` sets `abort_on_error=True`) or is retried, per the
   same policy. Tasks that exhaust retries become `IncompleteRange` entries
   (see [Resume and incomplete shots](#resume-and-incomplete-shots)).

### Chunk accumulation

Each completed task writes its own `chunk_<batch_id:06d>_<task_id[:8]>.parquet`
under `chunks/batch=<N>/`. Once the number of pending (unmerged) chunk files
reaches `--max_chunk_files` (default `1000`; `<=0` disables merging), groups of
`--merge_chunk_group_size` (default `None` → `min(200, max_chunk_files)`) are
concatenated into `chunks/batch=<N>/parts/part_<index:06d>.parquet` via
`polars.scan_parquet(...).sink_parquet(...)`, and the source chunks are
deleted. `collect_results` (final output-file grammar in
[ARCHITECTURE.md](ARCHITECTURE.md#the-result-collection-path)) reads both
`chunk_*.parquet` and `parts/part_*.parquet` when building final outputs.

### `threads × num_workers` (ILP only)

For `decoder: ILP`, `decoder_options.threads` (per-solve thread count) times
`--num_workers` (concurrent solves) is checked against the machine's available
core count and rejected with `ValueError` if it would oversubscribe — checked
identically for both solvers even though only Gurobi actually shares threads
across one process; CPLEX's `--num_workers` are separate OS processes.

### Atomic writes

Every final-output write (`logicalerror_batch=N.parquet`,
`metric=<name>_batch=N.parquet`, `detailed_stats_batch=N.parquet`,
`decoder_stat_batch=N.parquet`, `metadata_batch=N.json`,
`incomplete_shots.json`) goes through a temp-file-then-`os.replace()` pattern
(`_atomic_parquet` in `result_collection.py`; equivalent inline code in
`metadata.py`/`incomplete.py`), so concurrent batch-level writers to the same
directory never observe a partially-written file. Each batch also writes to
its **own** `metadata_batch=N.json`/output files rather than a shared path, so
there is no cross-batch overwrite in the first place.

### Relationship to `documents/archive/parallelization_report.md` and `parallel_execution_problem.md`

Both documents describe **already-fixed** issues, and both fixes are present
in the current code (verified): `parallelization_report.md`'s double-spawn fix
(a separate 1-worker probe pool, then a 96-worker main pool) is exactly the
"run probe inside the already-initialised pool" comment at
`execution/manager.py:132`; `parallel_execution_problem.md`'s NFS race fix is
the atomic-write pattern above. Neither document describes or anticipates the
probe-hang timeout gap (issue #2) — that gap is in the code they both left
behind, not something they claim to have addressed. See [FUTURE.md](FUTURE.md) for the
current, still-open state of that gap.

## Resume and incomplete shots

If any tasks are abandoned (timeout) or exhaust their retries (error) during a
`decoding` run, `outcome.incomplete` is non-empty: `incomplete_shots.json` is
written under the batch's output directory (schema: `schema_version`,
`ranges` — each with `shot_id_start`/`shot_id_end`/`num_shots`/`batch_id`/
`dets_path`/`reason` (`"timeout"`|`"error"`)/`message`/`attempts` —, and
`total_incomplete_shots`), and `main()` raises `IncompleteTasksError` (a
`RuntimeError` subclass) **after** writing the completed shots' outputs — a
failed run still leaves usable partial data plus a precise account of what's
missing. If `outcome.incomplete` is empty but a stale `incomplete_shots.json`
exists from a previous attempt, it is deleted.

`python -m decoder_confidence.decoding.resume` re-scans an interrupted output
directory's chunk/part parquets to determine which `shot_id` ranges are
already covered, builds tasks for only the gaps, and re-runs the normal
collect/finalize pipeline once they complete:

| Flag | Type | Required | Default |
|---|---|---|---|
| `--output_dir` | str | yes | — (the interrupted `decoding_result/decoder=X,metric=Y` dir) |
| `--decoder_config` | str | yes | — (same file as the original run) |
| `--batch_num` | int | no | auto-detected if exactly one `chunks/batch=N/` exists |
| `--num_workers` | int | yes | — |
| `--verbose` | bool | no | `False` |
| `--max_chunk_files` | int | no | `1000` |
| `--merge_chunk_group_size` | int | no | `None` |
| `--cleanup_intermediate` | bool | no | `True` |
| `--timeout_multiplier` | float | no | `10.0` |
| `--min_task_timeout_s` | float | no | `60.0` |
| `--max_task_retries` | int | no | `1` |
| `--maxtasksperchild` | int | no | `50` |

Like the main CLI, `resume.py` routes to a threaded or multiprocessing path by
the same `isinstance(decoder_factory, SharedEnvDecoderFactory)` check, and its
non-threaded (e.g. CPLEX) path re-merges the probe into the main pool for the
same reason — with the same probe-hang caveat as [Parallel execution](#parallel-execution).

## Cluster execution

The `*_cluster.sh` scripts assume a **PBS/Torque scheduler** (confirmed:
`#PBS -N`/`#PBS -l select=1:ncpus=...`/`#PBS -o`/`#PBS -e` directives, job
submitted with `qsub`). Each script builds a heredoc job script and submits it;
before use, edit the variables at the top of the file — at minimum
`PBS_NCPUS`, `CONDA_ENV`, and the simulation parameters (`CODE`, `DATA_DIR`,
`D`, `P`, `NOISE_MODEL`, `DECODER_CONFIG`, etc.) — and keep `NUM_WORKERS` in
sync with `PBS_NCPUS` (scripts warn, but do not fail, if they differ).
`step3_collect_wills_reproduce_batches_cluster.sh` and
`forcing_degradation_collect_batches_cluster.sh` submit one job per batch
number in a `BATCH_NUMS` list — edit that list to match the `--num_batch` used
in the corresponding step-2 run.

Retrieving results:

- **`fetch_from_cluster.sh`** — edit `REMOTE_SOURCES` (paths/globs on the
  `cluster` SSH host) and `LOCAL_PATH`. It creates a remote `tar.gz` of exactly
  those sources over `ssh`, downloads it (via `scp.exe` under WSL when
  available, else plain `scp`), extracts locally under `LOCAL_PATH`, and
  cleans up both the remote and local temp archives. `fetch_cluster.sh` is an
  older, Japanese-commented version of the same idea with different
  `REMOTE_SOURCES` — prefer `fetch_from_cluster.sh`.
- **`extract_tar.sh`** — a separate, simpler script for a plain local `tar`
  archive (e.g. one produced by an on-cluster backup script, not by
  `fetch_from_cluster.sh`'s own transfer): edit `INPUT_TAR`/`OUTPUT_DIR` at the
  top, then run it. Preserves the archive's original absolute-path structure
  as a relative tree under `OUTPUT_DIR`.

## Analysis

`analysis/` is not an installable package — run its scripts as modules from
the **repository root** with the root on `PYTHONPATH` (`export
PYTHONPATH="$PWD"`, or `export PYTHONPATH="$PWD:$PWD/src"` if you also need
`decoder_confidence` directly):

```bash
python -m analysis.examples.plot_metric_distribution
```

`analysis/examples/` (`plot_metric_distribution.py`, `plot_postselect.py`,
`inspect_dem_hyperedges.py`, plus `example.ipynb`/`test.ipynb`) and
`analysis/circuit_level/` (`anchored_gap_conditional_stats.py`,
`circuit_level.ipynb`) point at `example_outdir` (repo-root relative) by
default — that directory is gitignored, so a fresh clone must populate it
first (run the [Quickstart](README.md#quickstart) or point the script at your own result
directory).

Every entry point ultimately goes through `analysis.src.data_manager.SimulationDataManager(result_dir_root)`,
which expects exactly the layout stage 3 produces:

```
result_dir_root/
└── <circuit_params>/            # e.g. code=surface_code_Z,d=5,p=0.001,...
    └── decoding_result/
        └── <decoder_params>/    # e.g. decoder=ILP,metric=logical_gap
            ├── metric=<name>_batch=<N>.parquet
            └── logicalerror_batch=<N>.parquet
```

`.query(config)` (a `PlotConfig` or `ConditionalLERConfig`) returns a lazy,
directory-scan-filtered, `shot_id`-joined `polars.LazyFrame`; `filters` (e.g.
`{"d": 5, "p": 0.001}`) restrict which circuit-parameter directories are
scanned, and `group_by` selects which columns split the result into separate
plotted series. `NumericMetricAnalyzer`/`BooleanMetricAnalyzer`
(`analysis/src/analyzers.py`) then dispatch on whether the metric's output is
numeric or boolean (README.md's [metrics table](README.md#supported-confidence-metrics)).

## Troubleshooting

| Symptom | Cause | Where |
|---|---|---|
| `FileNotFoundError: Circuit directory not found under ...` | No directory under `--data_dir` matches every supplied `--code`/`--d`/`--rounds`/`--noise_model`/`--p`/`--xyz`/`--ibm_reproduce` | `decoding/__main__.py:_find_circuit_dir` |
| `FileExistsError: Multiple circuit directories matched ...` | Two+ directories under `--data_dir` match the same parameters (e.g. leftover data from an earlier `--out_dir`) | same |
| `RuntimeError: <lib> is required for <decoder> adapter` | The decoder's optional dependency isn't installed | `decoding/_decoder_adapter.py` |
| `ValueError: Unsupported relay-bp decoder option(s): ...` | A `decoder_options` key isn't accepted by the installed `relay_bp.bp.RelayDecoderF64` signature | `decoding/_decoder_adapter.py:_build_relay_decoder` |
| `ValueError: decoder_options.solver is required for ILP decoder` | `decoder: ILP` config is missing `decoder_options.solver` | `decoding/decoder_factory.py` |
| `ValueError: num_threads (...) * num_workers (...) exceeds available cores (...)` | ILP `threads × --num_workers` oversubscribes the machine | `decoding/__main__.py`, `resume.py` |
| `IncompleteTasksError` after outputs are written | Some tasks timed out or errored past their retry budget | see [Resume and incomplete shots](#resume-and-incomplete-shots) — re-run via `decoding.resume` |
| A `decoding` run with `--num_workers >= 1` and a non-Gurobi/non-threaded decoder hangs indefinitely with no output, before any progress line is printed | The probe-hang gap in `execution/manager.py` — the initial probe task has no timeout protection | GitHub issue #2, see [FUTURE.md](FUTURE.md) |
| `pytest tests/` (no marker filter) errors on 3 `test_obs_flip_idx_e2e.py` tests | Real-Gurobi tests refuse to run without `RUN_GUROBI_TESTS=1` (intentional — see [README.md#testing](README.md#testing)) | `tests/conftest.py` |
| `ValueError: No metric columns found in chunk outputs` / `Configured metric '<m>' missing in chunk outputs` | `collect_results` found chunk parquets with no `metric_*` columns, or none matching the configured `metric:` | `decoding/result_collection.py` |
| `analysis.examples.*` scripts raise `FileNotFoundError` pointing at `example_outdir` | That directory is gitignored and absent on a fresh clone | run the Quickstart first, or point the script elsewhere |
