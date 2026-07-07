# Step 2: DEM Generation and Detector Sampling

This document describes `commands/step2_create_dem_and_sample.sh` and the module it
invokes, `python -m decoder_confidence.sampling` (entry point:
`src/decoder_confidence/sampling/__main__.py`). Step 2 is the second stage of the
simulation pipeline: given a pre-built Stim circuit (produced by step 1), it builds
the circuit's detector error model (DEM) and draws the random detector/observable
samples that step 3 (decoding) consumes.

Step 2 is idempotent: if `dem.dem` or a given `det_batch=N.b8` file already exists,
it is left untouched and a log line is emitted instead of overwriting it. This makes
it safe to re-run the same command to fill in missing batches without disturbing
data already on disk.

## Usage

```bash
"${PYTHON_BIN}" -m decoder_confidence.sampling \
    --code ${CODE} \
    --out_dir ${OUT_DIR} \
    --noise_model ${NOISE_MODEL} \
    --rounds ${ROUNDS} \
    --d ${D} \
    --p ${P} \
    --num_shots ${NUM_SHOTS} \
    --xyz_decoding ${XYZ_DECODING} \
    --det_sample_seed ${DET_SAMPLE_SEED} \
    --num_batch ${NUM_BATCH} \
    --sampling_method ${SAMPLING_METHOD}
```

`PYTHONPATH` must include `src/` (the shell script exports this itself).

## Arguments

| Flag | Type | Required | Meaning |
|---|---|---|---|
| `--code` | str | yes | Code family name, e.g. `hgp_code_Z`, `surface_code_Z`, `bivariate_bicycle_code_Z`. Used together with `--d`, `--rounds`, `--noise_model`, `--p` (and, for bivariate-bicycle codes, `--xyz_decoding`) to locate the matching `.stim` file under `circuits/` by parsing its filename (see [Circuit lookup](#1-circuit-lookup)). |
| `--out_dir` | str | yes | Root directory under which the output layout (see [Output layout](#output-layout)) is created. |
| `--noise_model` | str | yes | Noise-model token as encoded in the circuit filename, e.g. `uniform`, `phenomenological`. |
| `--rounds` | str | yes | Rounds token as encoded in the circuit filename. Kept as a string (not parsed as int) because some circuit filenames encode rounds symbolically. |
| `--d` | int | yes | Code distance. Must be ≥ 1. |
| `--p` | float | yes | Physical error rate / noise strength. Matched against the circuit filename with an absolute tolerance of `1e-12`. |
| `--num_shots` | int | yes | Total number of shots to sample across all batches combined. |
| `--det_sample_seed` | int | yes | Base seed for detector sampling. Must be in `[0, 2**64 - 1]`. How it is used depends on `--sampling_method` (see [Detector sampling methods](#detector-sampling-methods)). |
| `--num_batch` | int | yes | Number of batches to split `--num_shots` into. Must satisfy `1 <= num_batch <= num_shots`. Batch sizes are computed by `compute_batch_sizes`: `num_shots // num_batch`, with the first `num_shots % num_batch` batches getting one extra shot, so batch sizes never differ by more than 1. |
| `--xyz_decoding` | bool | no (default `False`) | If `True`, keep detectors for both the X and Z stabilizer bases (used by codes that support joint XYZ decoding). If `False`, the circuit/DEM is restricted to a single basis (see [DEM generation and basis filtering](#2-dem-generation-and-basis-filtering)). Accepts `true/false/1/0/yes/no/t/f/y/n` (case-insensitive). |
| `--sampling_method` | str (choice) | no (default `unified`) | Detector-sampling strategy: `unified` or `per_batch_seed`. See [Detector sampling methods](#detector-sampling-methods) below — this is the option this document was added to explain in detail. |

## Internal pipeline

`main()` in `__main__.py` performs, in order:

### 1. Circuit lookup

`find_circuit_file()` scans `circuits/` recursively for a `.stim` file whose name
parses (via `_parse_circuit_stem`) to the same `code`, `d`, `rounds`, `noise_model`,
and `p` as the CLI arguments. For `bivariate_bicycle_code*`, the filename's
`use_both` flag must also match `xyz_decoding` (`use_both=True` ↔
`xyz_decoding=True`; the circuit already exists pre-built for both settings). It is
an error if zero or more than one file matches.

### 2. DEM generation and basis filtering

`_needs_dem_filter(code, xyz_decoding)` decides whether the DEM must be filtered to
a single stabilizer basis after generation:

- **`surface_code*` with `xyz_decoding=False`**: `True`. The full DEM is built via
  `circuit.detector_error_model(decompose_errors=False)`, then
  `filter_dem_by_basis()` removes all detectors of the basis identified by
  `_get_remove_basis(code)` (a code name ending in `_Z` keeps Z detectors and
  removes X; `_X` keeps X and removes Z). Detector IDs are remapped to a contiguous
  range afterward.
- **Everything else** (`xyz_decoding=True` for any code, or
  `bivariate_bicycle_code` with `xyz_decoding=False`): `False`. A standard DEM is
  generated directly via `generate_dem()`. Bivariate-bicycle codes already use a
  pre-built single-basis circuit file (`use_both=False` in the filename) when
  `xyz_decoding=False`, so no post-hoc filtering is needed.
- `superdense_color_code` basis filtering is not yet implemented (see the `TODO`
  in `__main__.py`).

The resulting DEM is written to `dem.dem` (skipped if it already exists).

### 3. Sampling

The DEM just written is reloaded from disk
(`stim.DetectorErrorModel.from_file(...)`) and passed to `sample_batches_from_dem()`
together with `batch_sizes`, `det_sample_seed`, and `sampling_method`. Reloading
from disk (rather than reusing the in-memory DEM) guarantees the sampled syndrome
data is always consistent with the exact DEM file that step 3 (decoding) will later
read.

Each batch is written as `sampled_data/det_batch={batch_index}.b8` (`batch_index`
starting at 1), containing, per shot, the detector bits followed by the observable
bits, packed per Stim's `b8` format (little-endian bit order within each byte,
zero-padded to a whole number of bytes per shot).

### 4. Metadata

`build_metadata()` collects the resolved configuration (including `sampling_method`)
plus circuit/output paths and the computed `batch_sizes`, and `write_metadata()`
serializes it to `metadata.json`.

## Detector sampling methods

`--sampling_method` selects between two mutually exclusive strategies for turning
`--num_shots` + `--det_sample_seed` + `--num_batch` into per-batch sample files.
Both are implemented in `sample_batches_from_dem()` / `sample_batches()`
(`src/decoder_confidence/sampling/sampler.py`); the constant
`SAMPLING_METHODS = ("unified", "per_batch_seed")` is the single source of truth for
the accepted values.

### `unified` (default, current/recommended)

All `num_shots` are drawn in **one** call, seeded once with `det_sample_seed`:

```python
sampler = dem.compile_sampler(seed=det_sample_seed)
dets, obs, _ = sampler.sample(total_shots)
```

The resulting array is then sliced deterministically into batches according to
`batch_sizes` (batch 1 gets rows `[0, batch_sizes[0])`, batch 2 gets the next
`batch_sizes[1]` rows, and so on) and each slice is written to its own
`det_batch=N.b8` file.

**Property:** the sampled shot set depends only on `num_shots` and
`det_sample_seed` — it is completely independent of how those shots are divided
into batches. Splitting 1,000,000 shots into 10 batches of 100,000 or 4 batches of
250,000 (same `det_sample_seed`) yields the exact same underlying 1,000,000-shot
sample, just partitioned differently across files. This is covered by
`test_dem_sampling_is_independent_of_batch_partition` /
`test_circuit_sampling_is_independent_of_batch_partition` in
`tests/test_sampling_batches.py`.

### `per_batch_seed` (legacy, pre-2026-06-25 behavior)

One sampler is compiled **per batch**, with a seed derived from the batch index:

```python
seed = det_sample_seed + (batch_index - 1)   # batch_index starts at 1
sampler = dem.compile_sampler(seed=seed)
dets, obs, _ = sampler.sample(batch_shots)
```

**Property:** the sampled shot set depends on `num_batch`, even when `num_shots`
and `det_sample_seed` are unchanged, because changing `num_batch` changes both the
per-batch shot counts and which seed each batch uses. Two runs with the same
`det_sample_seed` but a different `num_batch` are **not** reproductions of one
another under this method — this is exactly the property `unified` was introduced
to fix. This is covered by `test_dem_per_batch_seed_depends_on_batch_partition` in
`tests/test_sampling_batches.py`.

`seed = det_sample_seed + (batch_index - 1)` is validated to stay within
`[0, 2**64 - 1]`; a `det_sample_seed` close to `2**64 - 1` combined with a large
`num_batch` can overflow this bound and raises `ValueError`.

### When to use which

- Use `unified` (the default) for all new simulation runs. It is the only method
  for which the sampled data is reproducible independent of batching/parallelism
  choices, and is what `commands/step2_create_dem_and_sample.sh` sets explicitly.
- Use `per_batch_seed` only to reproduce, or directly compare against, data that
  was generated before the fix (i.e. `sampled_data/` directories whose
  `metadata.json` predates this option / has no `sampling_method` field, or whose
  file timestamps predate 2026-06-25). It should not be used for new production
  data.

`metadata.json` now always records which method was used
(`"sampling_method": "unified" | "per_batch_seed"`), so any future investigation of
an existing `sampled_data/` directory can read this field directly instead of
inferring it from file timestamps.

## Output layout

For `out_dir=<OUT>`, `circuit_id=<STEM>` (the matched circuit file's stem), and
`xyz_decoding=<XYZ>`:

```
<OUT>/<STEM>,xyz=<XYZ>/
├── dem.dem                        # DetectorErrorModel, possibly basis-filtered
├── metadata.json                  # resolved config + batch_sizes + sampling_method
└── sampled_data/
    ├── det_batch=1.b8             # detector bits ++ observable bits, per shot
    ├── det_batch=2.b8
    └── ...
```

This is exactly the layout that step 3 (decoding) and the `analysis/` package
expect: `dem.dem` for building/validating decoders, `sampled_data/det_batch=N.b8`
as the per-batch syndrome input, and `metadata.json` for provenance (including,
now, which sampling method produced the data).
