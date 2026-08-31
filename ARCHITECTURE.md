# ARCHITECTURE

How the code is put together, for a contributor or an agent that needs to
modify it. For how to *run* things, see [USAGE.md](USAGE.md); for what's broken or missing,
see [FUTURE.md](FUTURE.md).

## Data flow

```
 circuit (.stim)                    stage 1, external
       │
       ▼
 DEM (dem.dem)             stage 2  decoder_confidence.sampling
       │                            (detector_error_model(), optional
       │                             filter_dem_by_basis())
       ▼
 detector batches                   sampled_data/det_batch=N.b8
 (det_batch=N.b8)                   (unified or per_batch_seed, §Determinism)
       │
       ▼
 decoder, per shot         stage 3  decoder_confidence.decoding
       │                            (worker.py:run_task → DecoderBase.decode())
       ▼
 per-shot metric records            chunk_XXXXXX_xxxxxxxx.parquet
       │                            (§Result-collection path)
       ▼
 aggregated parquet                 metric=<name>_batch=N.parquet,
                                     logicalerror_batch=N.parquet, ...
       │
       ▼
 analysis                  stage 4  analysis.src.SimulationDataManager
                                     → NumericMetricAnalyzer /
                                       BooleanMetricAnalyzer / ...
```

## Module responsibilities

### `src/decoder_confidence/`

| Module | Responsibility |
|---|---|
| `config.py` | `SamplingConfig`/`DecodingResult` dataclasses; stage-2 `parse_args`; `compute_batch_sizes` |
| `varint.py` | VarInt encoding for `obs_flip_idx` binary files |
| `cli.py`, `__init__.py` | Empty |
| `sampling/__main__.py` | Stage-2 CLI: circuit lookup → DEM build/filter → sampling |
| `sampling/dem.py` | `OutputLayout`, `find_circuit_file`, `generate_dem`, `filter_dem_by_basis`, metadata build/write |
| `sampling/sampler.py` | `sample_batches_from_dem` — `unified`/`per_batch_seed` strategies, `.b8` writer |
| `decoding/__main__.py` | Stage-3 CLI: arg parsing, `_find_circuit_dir`, one-batch orchestration |
| `decoding/decoder_factory.py` | YAML → `(DecoderFactory, DecoderConfigInfo)`; the `(decoder, metric)` dispatch registry |
| `decoding/_decoder_adapter.py` | `DecoderAdapter` ABC + concrete adapters (ILP, BP-LSD, MWPM, RELAY-BP) + `build_decoder_adapter` dispatcher; DEM→matrix conversion |
| `decoding/_constraints.py` | Constrained-decode helpers (force a logical flip; random-split partitioning) |
| `decoding/_linearize_logicalgap.py` | `linearize_logicalgap` metric; also hosts shared helpers (`_get_obs_row`, `_append_row`, `_logical_from_correction`, `_is_obs_flip`) imported by four other metric modules |
| `decoding/_forced_gap.py` | `forced_gap_ml` metric (reselecting two-stage decode) |
| `decoding/_reweighted_linearized_gap.py` | `reweighted_linearized_gap` metric |
| `decoding/_argument_reweighting.py` | `ar-pec`/`ar-lec` metrics |
| `decoding/_wills_reproduce.py` | `wills_reproduce` metric — fixed-parameter reproduction of arXiv:2605.20346 |
| `decoding/_forcing_degradation_test.py` | `forcing_degradation_test` metric (BP-LSD/RELAY-BP only) |
| `decoding/_lsd_cluster_metric.py` | `cluster_llr` metric for BP-LSD; `_compute_cluster_llr`, reused by VIBE-LSD's variant |
| `decoding/_vibelsd.py` | VIBE-LSD ensemble decoder (BP ensemble + LSD fallback) and its `cluster_llr` variant |
| `decoding/_ilp_logicalgap.py` | Exact logical-gap ILP decoder, Gurobi backend; `SharedEnvDecoderFactory` |
| `decoding/_ilp_cplex_logicalgap.py` | Exact logical-gap ILP decoder, CPLEX backend; plain callable factory |
| `decoding/result_collection.py` | Chunk/part parquet → final per-batch outputs; atomic writes; legacy conversion |
| `decoding/metadata.py` | Per-batch `metadata_batch=N.json` build/write |
| `decoding/incomplete.py` | `incomplete_shots.json` read/write |
| `decoding/resume.py` | Resume an interrupted batch (threaded and non-threaded paths) |
| `decoding/forcing_degradation_collect.py` | Standalone collector for `forcing_degradation_test`, reusing `__main__`'s arg-parsing/dir-lookup |
| `execution/models.py` | `DecoderBase`, `DecoderFactory`, `SharedEnvDecoderFactory`, `SimulationTask`/`WorkerConfig`/`WorkerResult`/`ExecutionConfig`/`RunOutcome`/`IncompleteRange`/`IncompleteTasksError` |
| `execution/manager.py` | Routes to threaded or multiprocessing executor; multiprocessing orchestration |
| `execution/pool_runner.py` | `run_pool_tasks` — timeout/retry/hang detection over `apply_async` |
| `execution/threaded_manager.py` | `ThreadPoolExecutor`-based manager for `SharedEnvDecoderFactory` decoders |
| `execution/worker.py` | Per-process `init_worker`/`run_task`; decoder built once per worker |
| `execution/batching.py` | `estimate_shots_per_task`, `estimate_task_timeout_s` |
| `execution/hashing.py` | `stable_task_id`, `chunk_filename` |
| `execution/_execution_utils.py` | File discovery, core-affinity selection, part merging, RSS reporting |

### `analysis/src/`

| Module | Responsibility |
|---|---|
| `data_manager.py` | `SimulationDataManager` — walks the output-directory grammar, parses `key=value` dir names |
| `config.py` | `PlotConfig`/`ConditionalLERConfig`, metric-name normalization, `CONDITIONAL_LER_SUPPORTED_METRICS` |
| `analyzers.py` | `AbstractMetricAnalyzer` → `NumericMetricAnalyzer`/`BooleanMetricAnalyzer`; `ConditionalLERAnalyzer`; `PostselectionResult` |
| `postselect.py` | Post-selection performance plots |
| `case_histogram.py` | Histogram analysis for `forced_gap_ml_case` distributions |
| `anchored_reselection.py` | Diagnostics for anchored vs. reselecting forced-gap behavior |
| `ranking_degradation.py` | Ranking-degradation analysis (Wilson-CI based) |
| `stage_gap_difference.py` | Stage-1/stage-2 gap-difference statistics |
| `forcing_degradation.py` | Loader/config for forcing-degradation results |
| `relative_improvement.py` | Relative-improvement post-selection plot |
| `metric_correlation.py` | Shot-level Concordance Correlation Coefficient + scatter plots between metrics |
| `metric_spearman.py` | Spearman rank-correlation analysis |
| `split_analysis.py` | Split-parameter (random-split) analysis helpers |
| `confidence.py` | Wilson confidence-interval utilities |
| `figure_style.py` | Shared matplotlib styling |
| `plot_utils.py` | Notebook-facing plot/finalize helpers |
| `utils/length4_cycle.py` | Length-4-cycle diagnostic for constrained decode/random-split |

## The decoder abstraction

`config.DecodingResult` is the return value every decoder ultimately produces:
`predictions` (bool array), `metrics` (dict, becomes `metric_<key>` parquet
columns — every key except those starting with `__`, which are private/
consumed internally, e.g. `__is_logical_error` overrides the
`is_logical_error` output column), `detail_stats` (→ `detailed_stats_batch=N.parquet`
when `get_detail_stat=True`), `decoder_stats` (→ `decoder_stat_batch=N.parquet`),
and `obs_flip_idx` (→ `obs_flip_idx_batch=N.bin`).

`execution.models.DecoderFactory = Callable[..., DecoderBase]` is the
ordinary factory protocol: a zero/one-arg callable that returns a
`DecoderBase` (single abstract method: `decode(syndromes) -> DecodingResult`).
`SharedEnvDecoderFactory` (ABC) is the alternative for session-limited
decoders: `build_env()` is called exactly once, `build_decoder(env, dem)`
once per worker thread, and `__call__` is a single-shot fallback so it still
satisfies the plain `DecoderFactory` protocol where needed. The only current
implementer is Gurobi-backed ILP (`_ilp_logicalgap.py:_ILPDecoderFactory`).

`decoding/_decoder_adapter.py` is the shared layer beneath most metric
modules: `DecoderAdapter` (ABC: `priors`, `check_matrix`,
`observables_matrix`, `num_errors`, `decode`, `set_priors`,
`set_check_matrix`, `set_observables`) with concrete adapters
`IlpDecoderAdapter`, `BpLsdDecoderAdapter`, `MwpmDecoderAdapter`,
`RelayBpDecoderAdapter` (VIBE-LSD's adapter lives in `_vibelsd.py`, imported
lazily). `build_decoder_adapter(decoder_name, dem, decoder_options)`
dispatches on a normalized decoder name (`strip().upper().replace("_","-")`)
to the matching adapter builder, raising `ValueError(f"Unsupported base
decoder for AR: {decoder_name}")` for anything else — the message is a
naming leftover from when this dispatcher only served argument-reweighting;
it is now shared by six metrics.

A metric module composes an adapter into a factory in a consistent shape:
a frozen `*Options` dataclass plus a `_parse_*_options(metric_options) ->
Options` validator; a `*Decoder(DecoderBase)` class holding a
`DecoderAdapter` (obtained via `build_decoder_adapter(self.base_decoder, dem,
self.decoder_options)`) and implementing `decode()`; and a `make_*_factory(...)
-> DecoderFactory` that returns a closure/callable constructing that
`*Decoder`. `forced_gap_ml`, `linearize_logicalgap`,
`reweighted_linearized_gap`, `ar-pec`/`ar-lec`, and `wills_reproduce` all
follow this shape and therefore inherit `build_decoder_adapter`'s five
supported decoders (ILP, BP-LSD, MWPM, VIBE-LSD, RELAY-BP) without validating
the decoder themselves. `cluster_llr` and `forcing_degradation_test` instead
validate their (narrower) decoder sets explicitly, and ILP/VIBE-LSD as
*decoders* build their own factory directly rather than going through a
metric module (see dispatch order below).

### Dispatch order (`decoder_factory.py:load_decoder_factory`)

Dispatch is **metric-name-first**; order matters because later branches are
shadowed:

1. `metric == forcing_degradation_test` → validated decoder set `{BP-LSD, RELAY-BP}`
2. `metric ∈ {ar-pec, ar_lec, ar-lec, ar_pec}` → decoder unchecked here (delegated to `build_decoder_adapter`)
3. `metric == forced_gap_ml` → decoder unchecked here
4. `metric == wills_reproduce` → decoder unchecked here
5. `metric == reweighted_linearized_gap` → decoder unchecked here
6. `metric == linearize_logicalgap` → decoder unchecked here
7. `metric == cluster_llr` and `decoder ∈ {VIBE-LSD, VIBELSD}` → VIBE-LSD's own cluster_llr factory
8. `metric == cluster_llr` (else) → validated decoder set `{BP-LSD, BPLSD}`
9. `decoder == ILP` (metric matched none of 1–8) → `decoder_options.solver` (`gurobi`|`cplex`) selects the backend; **metric name is not checked at all**
10. `decoder ∈ {VIBE-LSD, VIBELSD}` (metric matched none of 1–8) → VIBE-LSD's own factory; **metric name is not checked at all**
11. else → `ValueError("Unsupported decoder/metric combination: decoder={decoder}, metric={metric}")`

Branches 9–10 are the source of the metric-validation gap recorded in
[FUTURE.md](FUTURE.md): an unrecognized `metric:` value combined with `decoder: ILP` or
`decoder: VIBE-LSD` is silently accepted and runs that decoder's own
fixed-metric computation, ignoring the requested (bogus) metric name entirely.

**Design note — `cluster_llr` via BP-LSD:** `ldpc`'s `LsdDecoder` exposes no
cluster-statistics API, so the `cluster_llr` metric (both the BP-LSD path in
`_lsd_cluster_metric.py` and the VIBE-LSD path in `_vibelsd.py`) is instead
built on `BpLsdDecoder` with `always_run_lsd=True` (forced regardless of
`decoder_options`) and `set_do_stats(True)`, so LSD's cluster stats are
populated on every shot even when BP itself converges — this is the only
practical way to get cluster statistics within `ldpc`'s current Python API.
`max_iter` and the rest of `decoder_options` remain user-configurable (not
forced) — an earlier design note (`Implementation_note.md`, now archived)
described `max_iter` as hardcoded to `1`; that is no longer the case in the
current code, verified against `_lsd_cluster_metric.py:130-141`.

## The execution model

`execution.manager.run_manager(config)` is the single entry point used by
both `decoding/__main__.py` and `decoding/resume.py`. It routes on
`isinstance(config.decoder_factory, SharedEnvDecoderFactory)`:

- **True** (Gurobi-backed ILP) → `threaded_manager.run_manager_threaded`: one
  shared Gurobi environment, a `ThreadPoolExecutor` with `num_workers`
  threads, each calling `build_decoder(env, dem)` once and reusing it across
  every task the thread handles.
- **False** (everything else, including CPLEX-backed ILP) →
  `manager._run_manager_multiprocess`: a `multiprocessing.Pool` using the
  `spawn` context, `initializer=init_worker`.

**Why decoder construction happens inside the worker:** `init_worker`
(the `Pool` initializer, called once per worker *process* at pool-spawn
time — not once per task) loads the DEM, builds the decoder via the supplied
`DecoderFactory`, and stashes it in a module-global `WorkerState`; `run_task`
then reuses that already-built decoder for every task the process is handed.
Building a decoder (parsing a DEM into check matrices, constructing a
Gurobi/CPLEX model, compiling a BP schedule, etc.) is expensive relative to
decoding one task's shots, so amortizing it once per process — instead of
once per task — is the entire point of `maxtasksperchild` being the process
*recycling* interval rather than 1. `init_worker` catches and stores any
construction exception in `_INIT_ERROR` rather than letting it propagate and
have `multiprocessing.Pool` silently and endlessly respawn workers; the first
`run_task` call in that state returns `WorkerResult(status="error", ...)`
carrying the traceback.

**`spawn` context:** used explicitly (`mp.get_context("spawn")`) rather than
the platform default (`fork` on Linux) so each worker starts from a clean
interpreter and re-imports every library fresh — required because several
optional decoder backends (Gurobi, CPLEX) hold C-level state (license
sessions, solver handles) that does not survive a `fork()` safely.

**Task granularity and hashing:** `execution.batching.estimate_shots_per_task`
sizes tasks from a timed probe; `execution.hashing.stable_task_id` hashes
`(decoder_name, dets_path, start_shot_index, num_shots, batch_id,
shot_id_offset)` into the deterministic chunk filename
`chunk_<batch_id:06d>_<task_id[:8]>.parquet`, so the same task always maps to
the same chunk path — the multiprocessing path's retry/collateral-resubmit
logic (`pool_runner.run_pool_tasks`) relies on this determinism to avoid
producing duplicate output for a task that gets resubmitted. Full mechanics,
including the probe-hang gap in step 1 of task setup, are in
[USAGE.md#parallel-execution](USAGE.md#parallel-execution).

## The result-collection path

Each completed task writes one chunk parquet
(`chunk_<batch_id:06d>_<task_id[:8]>.parquet`) under `chunks/batch=N/`. Once
pending chunks reach `--max_chunk_files`, groups of `--merge_chunk_group_size`
are concatenated into `chunks/batch=N/parts/part_<index:06d>.parquet`
(`_execution_utils.merge_chunks_into_part`, via `polars.scan_parquet(...).sink_parquet(...)`)
and the source chunks deleted.

`result_collection.collect_results(chunk_dir, output_dir, batch_num, metric)`
reads every `chunk_*.parquet` and `parts/part_*.parquet` (`pl.read_parquet`,
eager — a prior lazy/`scan_parquet`-re-evaluated version caused the NFS race
documented in `documents/parallel_execution_problem.md`, see [FUTURE.md](FUTURE.md)) and
writes, per batch:

| Output file | Contents |
|---|---|
| `logicalerror_batch=N.parquet` | `shot_id`, `is_logical_error` |
| `metric=<name>_batch=N.parquet` | one per `metric_*` column found, `shot_id` + value |
| `detailed_stats_batch=N.parquet` | if any `detail_*` columns exist — see rename map below |
| `decoder_stat_batch=N.parquet` | if any `decoder_*` columns exist |
| `obs_flip_idx_batch=N.bin` | if an `obs_flip_idx` column exists (VarInt-encoded, `varint.py`) |

**Detail-stats rename map** (`DETAIL_STAT_COLUMN_RENAMES`): the raw per-stage
column names (`stage1_obs_flip`, `stage1_weight`, `stage2_obs_flip`,
`stage2_weight`, `stage2_2ndbest_obs_flip`/`stage2_2nd_best_obs_flip`,
`stage2_2ndbest_weight`/`stage2_2nd_best_weight` — both a no-underscore and an
underscored spelling of "2nd best" are accepted) are renamed to
`baseline_logical_error`, `baseline_correction_weight`,
`forced_logical_error`, `forced_correction_weight`,
`forced_2nd_best_logical_error`, `forced_2nd_best_correction_weight` in the
output file.

**Legacy-conversion path** (`convert_legacy_detail_stats`): older runs wrote
one parquet file per detail-stat field (matched by
`metric=<legacy_name>_batch=*.parquet` or `<legacy_name>_batch=*.parquet`,
via `_LEGACY_DETAIL_BATCH_RE`) instead of one consolidated
`detailed_stats_batch=N.parquet`. This function detects such legacy files per
batch, joins them on `shot_id`, and writes the new consolidated format —
leaving the legacy files in place and skipping any batch that already has the
new-format file, so it is safe to call repeatedly and never overwrites a
just-finished batch with stale data.

All writes use `_atomic_parquet` (temp file + `os.replace()`), making
concurrent batch-level `collect_results` calls against the same
`decoding_result/...` directory safe — see [USAGE.md#parallel-execution](USAGE.md#parallel-execution).

## Determinism and seeding

`det_sample_seed` (plus `num_shots`, `num_batch`, and `sampling_method`)
fully determines the sampled detector/observable data — see
[documents/STEP2_SAMPLING.md](documents/STEP2_SAMPLING.md) for the full mechanics. In summary:

- **`sampling_method: unified`** (default): one `dem.compile_sampler(seed=det_sample_seed)`
  call draws all `num_shots` at once; the result is sliced deterministically
  into per-batch files. The sampled shot set is a pure function of
  `(det_sample_seed, num_shots)` — **independent of `num_batch`**. Splitting
  the same seed/shot-count into a different number of batches redistributes
  rows across files but never changes which shots were drawn.
- **`sampling_method: per_batch_seed`** (legacy): one sampler per batch, seeded
  `det_sample_seed + (batch_index - 1)`. The sampled shot set **depends on
  `num_batch`** — changing the batch count changes both the per-batch shot
  counts and which seed each batch uses, so two runs with the same
  `det_sample_seed` but different `num_batch` are not reproductions of one
  another. Kept only to reproduce/compare pre-2026-06-25 data.

Downstream, decoding is itself deterministic given its inputs (DEM, detector
bits, `decoder_options`/`metric_options`) for every decoder except where a
decoder's own algorithm is intentionally randomized by a `seed`
`decoder_option` (VIBE-LSD's permutation/perturbation sampling, RELAY-BP's
relay-leg exploration, `random_split`'s partition draw) — those are
deterministic *given* that seed, not seed-free.

## The `analysis/` layer

`analysis.src.data_manager.SimulationDataManager(result_dir_root)` walks
`result_dir_root/<circuit_params>/decoding_result/<decoder_params>/` (parsing
each `key=value,...` directory name via `_parse_kv`/`_is_circuit_params_dir`
and extracting the batch index from filenames via `_extract_batch_idx`), and
its `.query(config)` returns a lazily `shot_id`-joined `polars.LazyFrame`
across every matching directory and batch — no data is read until the caller
calls `.collect()`.

The analyzer hierarchy (`analyzers.py`) mirrors the numeric-vs-boolean split
that also shows up in README.md's metrics table: `AbstractMetricAnalyzer`
(ABC, one abstract method `plot_distribution(lf, config, ax)`) is implemented
by `NumericMetricAnalyzer` (scatter/histogram plots for continuous metrics
like `logical_gap`) and `BooleanMetricAnalyzer` (accept-rate bar charts for
`ar-pec`/`ar-lec`). `ConditionalLERAnalyzer` is a separate, non-`AbstractMetricAnalyzer`
class restricted to `CONDITIONAL_LER_SUPPORTED_METRICS = {"logical_gap",
"linearize_logicalgap", "forced_gap_ml"}` (`analysis/src/config.py`) — it
plots conditional logical-error rate against binned metric value, returning
`PostselectionResult` (`abort_rate`, `post_ler`, `original_ler`,
`reduction_rate`, `n_accepted`/`n_total`, Wilson `ci_low`/`ci_high`) for a
given post-selection threshold.
