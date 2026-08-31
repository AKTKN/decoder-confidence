# AGENTS

Behavioral contract for coding agents working in this repository. Not a
second architecture document — see [ARCHITECTURE.md](ARCHITECTURE.md) for how the code fits
together and [USAGE.md](USAGE.md) for how to run it.

## Project

`decoder-confidence` is a research pipeline for evaluating
**decoder-confidence metrics** as post-selection criteria in quantum error
correction: it decodes sampled syndrome data with several decoders
(ILP/Gurobi, ILP/CPLEX, BP-LSD, MWPM, VIBE-LSD, RELAY-BP) under several
confidence metrics (`logical_gap`, `cluster_llr`, `forced_gap_ml`,
`linearize_logicalgap`, `reweighted_linearized_gap`, `ar-pec`/`ar-lec`,
`wills_reproduce`, `forcing_degradation_test`), then analyzes which metrics
best predict a shot's actual logical error.

## Environment

- Python `>=3.10`; developed against the `decoder_confidence` conda
  environment (`conda activate decoder_confidence`; `environment.yml` is a
  snapshot of it).
- `PYTHONPATH` needs `src/` for `decoder_confidence.*`, and the **repository
  root** for anything under `analysis/` (no top-level `__init__.py` there —
  not an installable package).

## Commands

Safe default test suite (verified: 97 passed, 1 skipped):

```bash
export PYTHONPATH="$PWD/src"
python -m pytest tests/ -q -m "not e2e"
```

**Never** run `pytest tests/` or `-m e2e` without reading [README.md#testing](README.md#testing)
first — three tests call the real Gurobi API, rate/seat-limited on the
project's shared account; they error (not skip) unless `RUN_GUROBI_TESTS=1`.

Smoke-test sampling + decoding (verified end to end, smallest circuit, output
under `/tmp`):

```bash
export PYTHONPATH="$PWD/src"
python -m decoder_confidence.sampling --code surface_code_Z --out_dir /tmp/dc_smoke \
  --noise_model bitflip --rounds 1 --d 5 --p 0.1 \
  --num_shots 1000 --det_sample_seed 0 --num_batch 1 --xyz_decoding True
python -m decoder_confidence.decoding --code surface_code_Z --data_dir /tmp/dc_smoke \
  --noise_model bitflip --rounds 1 --d 5 --p 0.1 --batch_num 1 --num_workers 1 \
  --decoder_config conf/step3_config_cplex.yaml --xyz True
```

## Repository invariants — do not break these

- **Output directory-name grammar**: `_find_circuit_dir`/`_parse_dir_name`
  (`decoding/__main__.py`) parse `code=,d=,rounds=,noisemodel=,p=[,xyz=][,ibm_reproduce=True]`.
  Changing what sampling writes without updating the parser breaks
  discoverability of every existing result directory.
- **`shot_id` contract**: `shot_id = shot_id_offset + start_shot_index +
  local_index`, `shot_id_offset` being the cumulative shot count across
  `.b8` files in file order (`build_file_infos`) — a stable, globally unique
  key independent of task chunking. `collect_results` and
  `SimulationDataManager` join on it; resume relies on it being unambiguous.
- **Atomic writes**: every final-output write (`metric=*.parquet`,
  `logicalerror_*.parquet`, `detailed_stats_*.parquet`, `metadata_*.json`,
  `incomplete_shots.json`) must use the temp-file-then-`os.replace()`
  pattern — batches run concurrently against shared (often NFS) storage; a
  direct write is a race (see `documents/archive/parallel_execution_problem.md`).
- **`sampling_method: unified` determinism**: the sampled shot set must
  depend only on `(det_sample_seed, num_shots)`, never `num_batch` — the
  property `unified` exists to guarantee (`documents/STEP2_SAMPLING.md`).

## Conventions

- Metric/decoder implementation modules use a leading underscore
  (`_forced_gap.py`, `_decoder_adapter.py`, ...) — `decoder_factory.py` is
  the only intended caller from outside `decoding/`.
- Factory functions are named `make_<thing>_factory` and return a
  `DecoderFactory`-compatible callable.
- **Indentation is inconsistent — verified file-by-file, don't assume a
  rule.** Four-space is dominant (`analysis/src/`, `execution/`, almost all
  of `decoding/`, `sampling/__main__.py`). Tabs appear in exactly
  `config.py`, `decoding/_argument_reweighting.py`, `sampling/sampler.py`,
  and `sampling/dem.py` (itself mixed, mostly tabs). Match whichever a given
  file already uses.
- New metrics register a branch in `decoder_factory.py:load_decoder_factory`
  — the single dispatch point. Dispatch is metric-name-first and order
  matters (see [ARCHITECTURE.md#the-decoder-abstraction](ARCHITECTURE.md#the-decoder-abstraction)): an earlier branch
  matching your metric name shadows a later one.

## Prohibitions

- Never commit files under `.gitignore` paths (`Donotsync/`, `*.png`/`*.pdf`/`*.svg`,
  `raw_data/`, `simulation_data*/`, `simulation_result/`, `example_outdir/`,
  and other result directories — check `.gitignore` before adding anything
  under a circuit-parameter-shaped directory name).
- Never write simulation output into the repo tree outside those gitignored
  dirs — use `/tmp` or an explicit `--out_dir`/`--data_dir` for exploration.
- Never change the sampling seed logic (`sampling/sampler.py`,
  `SAMPLING_METHODS`) without an explicit migration note — `unified` exists
  because a prior scheme silently broke reproducibility across `num_batch`.
- Do not add a dependency without updating **both** `pyproject.toml` and
  `environment.yml` — see `FUTURE.md` §3 for a case this already slipped
  (CPLEX/`docplex`).

## Index

- Per-directory `AGENTS.md`: `src/decoder_confidence/sampling/`,
  `src/decoder_confidence/decoding/`, `src/decoder_confidence/execution/`,
  `analysis/src/`.
- Other documents: [README.md](README.md) (overview, quickstart), [USAGE.md](USAGE.md) (operational
  manual), [ARCHITECTURE.md](ARCHITECTURE.md), [documents/README.md](documents/README.md) (index of `documents/`).

## Known issues

`FUTURE.md` records known bugs, incomplete implementations, technical debt,
test-coverage gaps, and open questions. Read it before touching adjacent
code — **do not silently fix something recorded there as a side effect of
unrelated work**; if you do fix one, say so and update the entry.
