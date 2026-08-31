# AGENTS — `sampling/`

Stage 2: circuit lookup, DEM construction/filtering, detector sampling. See
the root [AGENTS.md](../../../AGENTS.md) and `documents/STEP2_SAMPLING.md` (authoritative for this
stage) first.

## Modules

| Module | Responsibility |
|---|---|
| `__main__.py` | CLI entry (`python -m decoder_confidence.sampling`): locate circuit → build/filter DEM → sample |
| `dem.py` | `OutputLayout`/`resolve_output_layout`, `find_circuit_file`, `generate_dem`, `filter_dem_by_basis`, metadata build/write |
| `sampler.py` | `sample_batches_from_dem` — `unified`/`per_batch_seed` strategies, `.b8` writer |
| `__init__.py` | Empty |

## Contract

- **Idempotence**: `generate_dem` and the per-batch `.b8` writer must skip
  (log, don't overwrite) if the target file already exists. Any change here
  must preserve that — downstream tooling re-runs the same sampling command
  to fill in gaps and relies on it being safe.
- **`unified` determinism**: the sampled shot set is a pure function of
  `(det_sample_seed, num_shots)`, independent of `num_batch` — see root
  `AGENTS.md`. Covered by
  `test_dem_sampling_is_independent_of_batch_partition` in
  `tests/test_sampling_batches.py`.
- `.b8` files pack detector bits then observable bits per shot,
  little-endian within each byte, zero-padded to a whole number of bytes
  (`bytes_per_shot = (num_detectors + num_observables + 7) // 8`) — this
  layout must match what `execution/_execution_utils.py:bytes_per_shot_count`
  and `worker.py:read_b8_slice` expect on the decoding side; changing it here
  without changing both breaks every batch file already on disk.

## Pitfalls

- `_needs_dem_filter` (`__main__.py`) unconditionally returns `False` for
  `superdense_color_code*` — requesting `--xyz_decoding False` for that code
  family silently produces a full both-basis DEM instead of erroring or
  filtering. See `FUTURE.md` §2.
- `dem.py` is internally mixed tab/space indented (verified: 123 tab lines,
  37 space lines) — match the surrounding block, don't reformat the whole
  file in an unrelated change.
- `find_circuit_file`'s directory/filename parsing (`_parse_circuit_stem`)
  and `decoding/__main__.py`'s `_parse_dir_name` are independent
  implementations of a similar grammar — a change to one's accepted keys
  (`use_both`, `ibm_reproduce`, `pcm`, ...) does not automatically apply to
  the other.

## Tests

`tests/test_sampling_batches.py` (both sampling methods, batch-partition
independence/dependence), `tests/test_e2e_pipeline.py::test_step2_sampling_xyz_suffix`
(e2e, output-layout check only, no decoding).
