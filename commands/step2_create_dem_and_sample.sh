#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR/src"

# surface_code_Z
"$PYTHON_BIN" -m decoder_confidence.sampling \
	--code bivariate_bicycle_code_Z \
	--out_dir example_outdir \
	--noise_model phenomenological \
	--rounds 6 \
	--d 6 \
	--p 0.02 \
    --num_shots 100 \
	--xyz_decoding False \
    --det_sample_seed 0 \
    --num_batch 1 \