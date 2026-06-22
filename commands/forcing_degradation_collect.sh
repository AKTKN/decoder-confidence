#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR/src"

# surface_code_Z
"$PYTHON_BIN" -m decoder_confidence.decoding.forcing_degradation_collect \
	--code surface_code_Z \
	--data_dir example_outdir \
	--noise_model bitflip \
	--rounds 1 \
	--d 5 \
	--p 0.1 \
	--xyz true \
	--batch_num 1 \
	--num_workers 4 \
	--decoder_config conf/forcing_degradation_bplsd.yaml \
	--verbose True
