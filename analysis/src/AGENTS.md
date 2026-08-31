# AGENTS — `analysis/src/`

Stage 4: load decoding results, plot metric distributions and
post-selection performance. See the root [AGENTS.md](../../AGENTS.md) and `ARCHITECTURE.md`'s
"The `analysis/` layer" first. (This file also serves as the intended
`analysis/implementation_guide.md`, which was empty and has been removed.)

## Modules

| Module | Responsibility |
|---|---|
| `data_manager.py` | `SimulationDataManager` — walks the output-directory grammar, `.query()` returns a lazy joined `LazyFrame` |
| `config.py` | `PlotConfig`/`ConditionalLERConfig`, metric-name normalization, `CONDITIONAL_LER_SUPPORTED_METRICS` |
| `analyzers.py` | `AbstractMetricAnalyzer` → `NumericMetricAnalyzer`/`BooleanMetricAnalyzer`; `ConditionalLERAnalyzer`; `PostselectionResult` |
| `postselect.py`, `relative_improvement.py` | Post-selection performance plots |
| `case_histogram.py`, `anchored_reselection.py`, `ranking_degradation.py`, `stage_gap_difference.py` | Diagnostic/case-breakdown analyses for the forced-gap family of metrics |
| `forcing_degradation.py` | Loader/config for forcing-degradation results |
| `metric_correlation.py`, `metric_spearman.py` | Cross-metric agreement (CCC, Spearman) |
| `split_analysis.py` | Random-split analysis helpers |
| `confidence.py` | Wilson confidence-interval utilities |
| `figure_style.py`, `plot_utils.py` | Shared matplotlib styling / notebook helpers |
| `utils/length4_cycle.py` | Length-4-cycle diagnostic; imports from `src/decoder_confidence` directly |

## Contract

- `analysis/` is **not** an installable package (no top-level `__init__.py`,
  no `pyproject.toml`). Every module here is only importable with the
  **repository root** on `PYTHONPATH`/`sys.path`.
- `SimulationDataManager(result_dir_root)` expects exactly stage 3's output
  layout: `result_dir_root/<circuit_params>/decoding_result/<decoder_params>/
  {metric=<name>_batch=N.parquet, logicalerror_batch=N.parquet}`. Don't
  change what stage 3 writes without checking `_parse_kv`/
  `_is_circuit_params_dir`/`_extract_batch_idx` still parse it.
- New numeric-metric analysis should go through `NumericMetricAnalyzer`
  (or `ConditionalLERAnalyzer` if the metric is added to
  `CONDITIONAL_LER_SUPPORTED_METRICS`); boolean/accept-reject metrics go
  through `BooleanMetricAnalyzer` — match README.md's metrics-table output
  type, don't guess.

## Pitfalls

- `utils/length4_cycle.py` inserts `src/` into `sys.path` directly
  (`REPO_ROOT`/`SRC_ROOT` computed from `__file__`) instead of relying on
  `PYTHONPATH` like every other module here — an inconsistency, not a
  template to copy.
- Several modules here are large (`case_histogram.py` 1513 lines,
  `analyzers.py` 1397, `stage_gap_difference.py` 1128, `postselect.py` 980,
  `anchored_reselection.py` 735) and were not read end-to-end while writing
  this documentation set — verify current behavior in the file itself
  before relying on a doc's summary of it.
- `analysis/examples/` and `analysis/circuit_level/` scripts default to
  pointing at `example_outdir` (repo-root relative), which is gitignored —
  they will raise `FileNotFoundError` on a fresh clone until it's populated.

## Tests

`test_metric_correlation.py`, `test_numeric_distribution.py`,
`test_override_gap_probability.py`, `test_length4_cycle_analysis.py`.
