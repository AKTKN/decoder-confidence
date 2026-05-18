#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR/src"

# surface_code_Z
"$PYTHON_BIN" -m decoder_confidence.sampling \
	--code surface_code_Z \
	--out_dir example_outdir \
	--noise_model bitflip \
	--rounds 1 \
	--d 5 \
	--p 0.1 \
    --num_shots 10000 \
	--xyz_decoding True \
    --det_sample_seed 0 \
    --num_batch 1 \