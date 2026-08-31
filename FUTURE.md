# FUTURE

Known shortcomings of the current implementation, so they aren't silently
"fixed" as a side effect of unrelated work. Do not treat anything here as
resolved without checking the cited location yourself first — some of it may
have moved on since this was written.

## 1. Known bugs and correctness concerns

Ordered by severity (most severe first).

### 1.1 Logical-gap computation may be silently wrong (unresolved, migrated from `Implementation_note.md`)

The repository's own prior investigation notes (now archived, see
`documents/archive/implementation_note.md`) record an unresolved correctness
problem with logical-gap computation. Translated from the original Japanese,
separating **confirmed observations** from **hypotheses**:

- **Confirmed (direct observation):** computed logical-gap values were
  "extremely small" — implausibly small relative to expectation.
- **Confirmed (direct observation):** ILP and MWPM sometimes disagreed on
  whether a given shot was a logical error, though they agreed in most cases.
- **Confirmed (asserted as a hard invariant violation):** negative gap values
  were observed; a negative gap should not be possible under the metric's
  definition.
- **Hypothesis:** `filter_dem_by_basis` (`sampling/dem.py`) was suspected as
  the root cause. Supporting evidence cited: inspecting a filtered DEM
  revealed an error mechanism that flips an observable (`L0`) with nonzero
  probability but touches **no detector** — i.e. an undetectable
  observable-flip mechanism, which should not survive a correct filtering
  step.
- **Hypothesis:** circuits going into step 2 were inconsistent about whether
  they carried both-basis or single-basis detectors, which was suspected to
  interact badly with `filter_dem_by_basis`.
- **Hypothesis (tentative, marked "↑" — a candidate explanation for the
  above, not confirmed):** the underlying cause may be in circuit
  construction rather than the filter itself — the two-qubit gate set (CZ)
  syndrome extraction may need explicit Hadamards before/after each CZ to
  correctly frame the basis transformation.

**Status:** `(unverified)` whether this is still reproducible in the current
code — confirming it would require running a decode and inspecting a
filtered DEM for zero-detector, nonzero-probability observable-flip error
mechanisms, which is outside the scope of a documentation pass. Treat
`logical_gap` (and any metric built on `filter_dem_by_basis`-filtered DEMs,
i.e. `surface_code*` with `xyz_decoding=False`) as unverified for
correctness until this is either reproduced or ruled out.

### 1.2 Multiprocessing probe task can hang indefinitely (GitHub issue #2, open)

In `_run_manager_multiprocess` (`execution/manager.py:136`) and the
non-threaded path of `resume.py` (`resume.py:570`), the initial probe task
runs via a **synchronous** `pool.apply(run_task, (probe_task,))` before
`run_pool_tasks`'s timeout/retry/termination machinery exists. If the probe
hangs (decoder deadlock, native-extension block, pathological input), the
parent process blocks forever — `task_timeout_s`/`max_task_retries` never
apply, because they're not in scope yet.

This affects every decoder **except** Gurobi-backed ILP (the only
`SharedEnvDecoderFactory`, which uses the threaded manager instead) —
including CPLEX-backed ILP, which is what every current `conf/*.yaml` ILP
config actually uses. Confirmed present, unmodified, as of this writing;
verified against the full commit history (only unrelated commits have
touched `manager.py`/`resume.py` since the issue was filed). See
[USAGE.md#parallel-execution](USAGE.md#parallel-execution) for the surrounding mechanics. Proposed fix (from
the issue): run the probe via `apply_async` with an explicit bounded
timeout, `pool.terminate()`/`join()` on timeout, and abort with a clear
error — mirroring the normal task path instead of bypassing it.

### 1.3 The main `decoding` CLI's per-task timeout is effectively unbounded

Even once past the probe (§1.2), `ExecutionConfig.timeout_multiplier`
defaults to `100000.0` (`execution/models.py:148`) and is **not exposed as a
CLI flag** on `python -m decoder_confidence.decoding` (only `resume.py`
exposes `--timeout_multiplier`, defaulting to a far more conservative
`10.0`). With a probe measuring, say, 1s/shot and a `shots_per_task` of 30,
the resulting timeout is on the order of `100000 * 30` seconds — in practice
no meaningful timeout at all on the primary entry point. A hung task on the
main CLI (as opposed to `resume.py`) will not be detected by this mechanism
regardless of §1.2's fix.

### 1.4 Metric name is not validated against the chosen decoder for `ILP`/`VIBE-LSD`

In `decoder_factory.py`, dispatch is metric-name-first (see
[ARCHITECTURE.md#the-decoder-abstraction](ARCHITECTURE.md#the-decoder-abstraction) for the full order). Branches 9–10
(`decoder == ILP` and `decoder ∈ {VIBE-LSD, VIBELSD}`) are reached whenever
`metric:` did not match any of the known metric names in branches 1–8 —
**and neither branch checks the metric name at all.** A misspelled or
unsupported `metric:` value combined with `decoder: ILP` or `decoder:
VIBE-LSD` is silently accepted; the run proceeds computing that decoder's own
fixed metric (`logical_gap` for ILP) while ignoring what was actually
requested in the config. No error, warning, or metadata field flags the
mismatch.

### 1.5 `conf/step3_config_linearize_bplsd.yaml` configures `cluster_llr`, not `linearize_logicalgap`

A concrete instance of the class of bug in §1.4's neighborhood (though this
one doesn't fall through dispatch — `cluster_llr` is itself a valid, checked
branch): the filename says "linearize" but `metric: cluster_llr`
(`conf/step3_config_linearize_bplsd.yaml:2`). Anyone selecting this file by
name expecting `linearize_logicalgap` output gets `cluster_llr` instead. See
[USAGE.md#configuration-files](USAGE.md#configuration-files).

### 1.6 `test_step3_decoding_e2e` gates on the wrong solver

`tests/test_e2e_pipeline.py:98` calls `pytest.importorskip("gurobipy")`
before running a decode that actually uses `conf/step3_config.yaml`, whose
`decoder_options.solver` is `cplex`, not `gurobi`. The guard is harmless in
its current form (gurobipy happens to be installed, so it never skips
anything, and no Gurobi API call is ever made by this test — it uses CPLEX)
but it is misleading to anyone auditing which tests touch which solver's API,
particularly next to the real Gurobi-API gate added in `tests/conftest.py`
(§4 test-coverage note below).

## 2. Incomplete implementations

- **`superdense_color_code` single-basis filtering is not implemented.**
  `_needs_dem_filter` (`sampling/__main__.py:25`) returns `False` for
  `superdense_color_code*` unconditionally, with a `TODO: enable once
  superdense_color_code pipeline is complete` at `sampling/__main__.py:38,94`.
  Practical effect: requesting `--xyz_decoding False` for this code family
  does **not** raise an error and does **not** silently do nothing obvious —
  it silently generates a **full, both-basis** DEM instead of the requested
  single-basis one, which could be mistaken for correct single-basis output
  if not checked against `metadata.json`.
- **`circuits/circuit_format.md` documents a circuit generator
  (`./scripts/generate_circuits`) that does not exist anywhere in this
  repository.** No `scripts/` directory exists. Circuit generation (stage 1)
  is external to this repo; this document appears to describe a wrapper that
  was planned but never added here, or that lives in a different, unlinked
  repository. `(unverified)` which.
- **`src/decoder_confidence/cli.py` is a completely empty file (0 bytes).**
  Its name suggests an intended unified CLI entry point, but nothing is
  implemented; both stages are actually invoked via `python -m
  decoder_confidence.sampling` / `python -m decoder_confidence.decoding`
  directly. `(unverified)` whether this is an abandoned stub or a
  placeholder for planned work — see Open questions.
- **API-surface inconsistency across package `__init__.py` files.**
  `decoding/__init__.py` and `execution/__init__.py` both re-export their
  package's public names via `__all__`; the top-level
  `src/decoder_confidence/__init__.py` and `sampling/__init__.py` are both
  empty (0 bytes) and re-export nothing.
- From `Implementation_note.md`'s "Future work" list (now archived): two
  items remain undone — downgrading metric storage from `float64` to
  `float32` to reduce memory (no evidence this was done — `metrics` arrays
  are constructed as plain `np.float64` throughout the metric modules
  surveyed), and "adaptive confidence calculation" (no metric module's
  behavior appears to adapt at runtime based on prior results;
  `(unverified)` whether anything in the codebase now satisfies this goal
  under a different name). The list's other two items — "add cluster
  metrics support" and "add heuristic logical gap calculation (by
  ensemble)" — are done (`cluster_llr`, VIBE-LSD respectively) and are not
  carried forward here.

## 3. Technical debt

- **CPLEX/`docplex` are undeclared dependencies.** Neither `pyproject.toml`
  nor `environment.yml` lists them, even though
  `decoding/_ilp_cplex_logicalgap.py` imports `ilp_decoder.cplex_backend`
  and `cplex_formulation` unconditionally at module level (reached only when
  `solver: cplex` is actually selected, so `import decoder_confidence`
  itself doesn't fail, but nothing records that a CPLEX install is needed).
  Both are, in fact, installed and importable in the environment this was
  written in — see [USAGE.md#prerequisites](USAGE.md#prerequisites).
- **`environment.yml`'s pinned versions for first-party packages have no
  installable source within that file.** `ilp-decoder==0.1.1` is listed with
  no VCS reference; `relay-bp` isn't listed at all (see below). This file is
  a `conda env export` snapshot, not a hand-maintained spec —
  `pyproject.toml`'s new `ilp`/`relay` extras (added this session, pinned to
  exact commits/tags) are the actual installable source for both.
- **Packaging asymmetry between `src/` and `analysis/`.** `src/decoder_confidence`
  is a proper installable package; `analysis/` has no `__init__.py` at its
  top level and no `pyproject.toml`/`setup.py` of its own — it's only
  importable with the repository root on `PYTHONPATH`. Some `analysis/`
  modules paper over this by inserting paths into `sys.path` directly at
  import time instead (e.g. `analysis/src/utils/length4_cycle.py:11-13`),
  which works but is inconsistent with how every other `analysis/` module
  expects to be imported.
- **Oversized modules** (line counts as of this writing): `analysis/src/case_histogram.py`
  (1513), `analysis/src/analyzers.py` (1397), `analysis/src/stage_gap_difference.py`
  (1128), `analysis/src/postselect.py` (980), `analysis/src/anchored_reselection.py`
  (735), `decoding/resume.py` (727), `decoding/_decoder_adapter.py` (696).
  None of these were read end-to-end for this documentation pass; their
  responsibility-table entries (`ARCHITECTURE.md`) are necessarily coarse.
- **`_linearize_logicalgap.py` is a de facto shared-helper module** despite
  its leading-underscore, single-metric-looking name: `_get_obs_row`,
  `_append_row`, `_logical_from_correction`, `_is_obs_flip`,
  `_normalize_true_obs` are imported from it by `_forced_gap.py`,
  `_reweighted_linearized_gap.py`, `_wills_reproduce.py`, and
  `_argument_reweighting.py`. A reader following the "leading underscore =
  private to this module" convention (see `AGENTS.md`) would not expect
  this.
- **`build_decoder_adapter`'s error message is a naming leftover.**
  `ValueError(f"Unsupported base decoder for AR: {decoder_name}")`
  (`_decoder_adapter.py:696`) still says "for AR" (argument-reweighting)
  even though the function is now the shared dispatcher for six metrics.
- **`conf/step3_config.yaml` and `conf/step3_config_cplex.yaml` are
  redundant** — byte-identical apart from a trailing newline (verified via
  `diff`). `(unverified)` whether the duplication is intentional (e.g. as a
  placeholder for the two to diverge later) or accidental.
- **`bench_cplex_5shots.py` sits at the repository root**, outside any
  package (`src/` or `analysis/`) — an ad hoc CPLEX-ILP benchmark script
  hardcoded to one specific BB-code circuit and 5 shots. It resembles the
  one-off scripts under `analysis/examples/` (e.g.
  `inspect_dem_hyperedges.py`) more than anything at the repository root.
  Recommend moving it under `analysis/` (e.g. `analysis/benchmarks/`) for
  consistency; not moved as part of this documentation-only pass.

## 4. Test coverage gaps

Full-repository search (`grep -rl`, zero false negatives for an exact
string): **no test file anywhere under `tests/` mentions `MWPM`,
`pymatching`, `VIBE-LSD`, or `VIBELSD`.** Both decoders have zero test
coverage of any kind — not even a mocked-adapter unit test.

| Decoder | Test coverage |
|---|---|
| `ILP` (Gurobi) | `tests/test_obs_flip_idx_e2e.py` only — `@pytest.mark.e2e` **and** `@pytest.mark.gurobi`, opt-in via `RUN_GUROBI_TESTS=1` (see `tests/conftest.py`). No fast/unit-level coverage. |
| `ILP` (CPLEX) | `tests/test_e2e_pipeline.py` only (`test_step3_decoding_e2e`, `test_step3_decoding_cplex_e2e`) — both `@pytest.mark.e2e`, run a real CPLEX solve. No fast/unit-level coverage. |
| `BP-LSD` | Exercised (not necessarily exhaustively) by several metric/analysis tests, e.g. `tests/test_forcing_degradation.py`, `tests/test_forced_gap_use_linearize.py`. |
| `MWPM` | **None.** |
| `VIBE-LSD` | **None.** |
| `RELAY-BP` | `tests/test_relay_bp_adapter.py`, `tests/test_relay_bp_nonconvergence.py`, `tests/test_wills_reproduce.py` — via a monkeypatched fake `relay_bp` module (`sys.modules` injection), so these run without the real package built. Good unit coverage of the adapter/dispatch logic; `(unverified)` whether any test exercises the *real* compiled `relay_bp` extension, which isn't currently built in this project's environments (see prior session investigation, not re-derived here). |

Per-metric: `ar-pec`/`ar-lec` (`test_argument_reweighting.py`, against a
`DummyAdapter`, not a real decoder), `wills_reproduce`
(`test_wills_reproduce.py`), and `forcing_degradation_test`
(`test_forcing_degradation.py`) each have a dedicated test file.
`forced_gap_ml`/`linearize_logicalgap` are covered together across
`test_forced_gap_use_linearize.py`, `test_detail_stat_metrics.py`, and
`test_override_gap_probability.py`. **`reweighted_linearized_gap` has no
dedicated test file** — the only match for it anywhere under `tests/` is
inside `test_relay_bp_nonconvergence.py`, which is scoped to non-convergence
handling, not the metric's core computation. `logical_gap` is covered only
by the e2e ILP tests above.

## 5. Open questions

- Should `wills_reproduce` validate `decoder: RELAY-BP` explicitly (the way
  `forcing_degradation_test` and `cluster_llr` do), given the metric's
  `decoder_options` (`num_sets`, `gamma0`, `pre_iter`, ...) are meaningless
  for any other decoder in `build_decoder_adapter`'s set? Currently a
  non-RELAY-BP decoder would presumably fail deep inside adapter
  construction with an unrelated error rather than a clear message at
  dispatch time — `(unverified)`, not reproduced.
- Is `src/decoder_confidence/cli.py` a live placeholder for planned work, or
  dead weight safe to remove?
- Is the `conf/step3_config.yaml` / `conf/step3_config_cplex.yaml`
  duplication intentional?
- `VibeLsdOptions.always_run_lsd` is explicitly documented as
  "internal — not parsed from YAML" (`_vibelsd.py:34`) and unconditionally
  popped from `decoder_options` before parsing. Is this meant to stay
  internal-only permanently, or become a real YAML-configurable option
  later?
- Is the logical-gap bug in §1.1 still current? The archived note's
  hypotheses were never confirmed or ruled out in the historical record
  available to this documentation pass.
