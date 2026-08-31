# AGENTS — `decoding/`

Stage 3: decoder/metric dispatch, per-metric decoders, result collection,
resume. See the root [AGENTS.md](../../../AGENTS.md) and `ARCHITECTURE.md`'s "The decoder
abstraction" first.

## Modules

| Module | Responsibility |
|---|---|
| `__main__.py` | CLI entry; `_find_circuit_dir` (output-directory grammar parser) |
| `decoder_factory.py` | `load_decoder_factory` — the single `(decoder, metric)` dispatch point |
| `_decoder_adapter.py` | `DecoderAdapter` + concrete adapters (ILP, BP-LSD, MWPM, RELAY-BP) + `build_decoder_adapter` |
| `_constraints.py` | Constrained-decode helpers (force a logical flip; random-split) |
| `_linearize_logicalgap.py`, `_forced_gap.py`, `_reweighted_linearized_gap.py`, `_argument_reweighting.py`, `_wills_reproduce.py`, `_forcing_degradation_test.py`, `_lsd_cluster_metric.py`, `_vibelsd.py`, `_ilp_logicalgap.py`, `_ilp_cplex_logicalgap.py` | One metric (or, for the ILP/VIBE-LSD pair, one decoder) each |
| `result_collection.py` | Chunk/part parquet → final outputs; atomic writes; legacy conversion |
| `metadata.py`, `incomplete.py` | Per-batch metadata / incomplete-shots manifests |
| `resume.py` | Resume an interrupted batch |
| `forcing_degradation_collect.py` | Standalone collector reusing `__main__`'s arg-parsing/dir-lookup |

## Contract

- Every metric decoder returns `config.DecodingResult(predictions, metrics,
  detail_stats, decoder_stats, obs_flip_idx)`. Every key in `metrics` except
  ones starting with `__` becomes its own `metric_<key>` parquet column
  (`worker.py:_normalize_metrics`) — don't add a `__`-prefixed key expecting
  it to show up as an output column.
- `load_decoder_factory` dispatch is **metric-name-first** and order
  matters: a new metric branch must go before the `decoder == ILP` /
  `decoder ∈ {VIBE-LSD, VIBELSD}` fallback branches, or it will never be
  reached for those decoders. See `ARCHITECTURE.md` for the full order.
- A new metric that reuses `build_decoder_adapter` (ILP, BP-LSD, MWPM,
  VIBE-LSD, RELAY-BP) gets those five decoders "for free" but does **not**
  get decoder validation — if your metric only makes sense for a subset,
  validate that yourself (see `forcing_degradation_test`'s explicit check as
  the pattern to follow).

## Pitfalls

- `decoder_factory.py`'s `decoder == ILP` / `VIBE-LSD` fallback branches
  never check the metric name at all — a typo'd `metric:` silently runs that
  decoder's fixed metric instead of erroring. `FUTURE.md` §1.4.
- `_linearize_logicalgap.py` is imported by four other metric modules for
  shared helpers (`_get_obs_row`, `_append_row`, `_logical_from_correction`,
  `_is_obs_flip`) despite its leading-underscore, single-metric-looking
  name — don't assume it's safe to change in isolation.
- `build_decoder_adapter`'s `ValueError` message says "for AR" — it's
  shared by six metrics now, the message is just stale (`FUTURE.md` §3).
- `conf/step3_config_linearize_bplsd.yaml` actually configures
  `cluster_llr`, not `linearize_logicalgap` — don't trust `conf/` filenames
  over their contents (`FUTURE.md` §1.5).

## Tests

`test_argument_reweighting.py` (AR, dummy adapter), `test_forced_gap_use_linearize.py`,
`test_detail_stat_metrics.py`, `test_override_gap_probability.py`
(`forced_gap_ml`/`linearize_logicalgap`), `test_wills_reproduce.py`,
`test_forcing_degradation.py`, `test_relay_bp_adapter.py`,
`test_relay_bp_nonconvergence.py` (RELAY-BP, via a monkeypatched fake
module), `test_incomplete_shots.py`. **No test file anywhere covers MWPM or
VIBE-LSD** (`FUTURE.md` §4). ILP is covered only by e2e tests
(`test_e2e_pipeline.py` for CPLEX, `test_obs_flip_idx_e2e.py` for
Gurobi — the latter gated behind `RUN_GUROBI_TESTS=1`, see root `AGENTS.md`).
