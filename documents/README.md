# documents/

Index of this directory. Root-level docs ([README.md](../README.md), [USAGE.md](../USAGE.md),
[ARCHITECTURE.md](../ARCHITECTURE.md), [FUTURE.md](../FUTURE.md), [AGENTS.md](../AGENTS.md)) are the primary reference set and
aren't repeated here.

## Current reference material

| Document | Covers |
|---|---|
| [STEP2_SAMPLING.md](STEP2_SAMPLING.md) | Authoritative reference for stage 2 (`decoder_confidence.sampling`): full CLI, internal pipeline, `--sampling_method` semantics, idempotence, output layout. `USAGE.md`'s Stage 2 section is a compact index into this document, not a duplicate. |
| [relay_bp_nonconvergence_behavior.md](relay_bp_nonconvergence_behavior.md) | How each decoder-independent metric (`linearize_logicalgap`, `forced_gap_ml`, `reweighted_linearized_gap`, `ar-pec`/`ar-lec`) handles RELAY-BP non-convergence, and the history of the dead-code module (`_relay_bp.py`, since deleted) this behavior was migrated out of. Japanese. |

## Archived records

Superseded or fully-resolved material, kept for history. Each file below
carries a header noting where its live content now lives.

| Document | Original subject | Status |
|---|---|---|
| [archive/implementation_note.md](archive/implementation_note.md) | Early design/bug notes (Japanese): DEM hyperedge decomposition, XYZ-decoding config, the analysis-module design, an unresolved logical-gap bug, a `cluster_llr`-via-BP-LSD design note | Split: bug → `FUTURE.md` §1.1, design note → `ARCHITECTURE.md`, "To do" items resolved, "Future work" items done or tracked in `FUTURE.md` |
| [archive/parallel_execution_problem.md](archive/parallel_execution_problem.md) | Bug report (Japanese) for a concurrent-batch NFS race in result/metadata writes | Fixed in current code; the atomic-write pattern it introduced is documented in `USAGE.md`/`ARCHITECTURE.md` |
| [archive/parallelization_report.md](archive/parallelization_report.md) | Diagnosis of a double-worker-spawn performance bug in the ILP/Gurobi execution path | Fix present in current code (`execution/manager.py` merges the probe into the main pool), documented in `ARCHITECTURE.md`'s "The execution model"; does not cover the still-open probe-timeout gap in `FUTURE.md` §1.2 |
| [archive/figure_planning_notes.md](archive/figure_planning_notes.md) | Informal Japanese notes on which figures to produce, formerly at the bottom of the root `README.md` | Retired; no longer reflects the current README's content |

## Not part of this documentation set

`analysis/relaybp_leranalysis.md` (a specific research finding on RELAY-BP
reselection-vs-anchoring behavior) and `parallel_conflict/REPRODUCTION_REPORT.md`
(inside a fully gitignored scratch directory) exist elsewhere in the
repository but are research artifacts, not reference documentation — they
aren't indexed here or in the root docs.
