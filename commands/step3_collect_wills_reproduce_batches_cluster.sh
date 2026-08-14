#!/usr/bin/env bash
# =============================================================================
# PBS Job Submitter — wills_reproduce: Step 3 (collect, batch submission)
#
# Submits commands/step3_collect_wills_reproduce_cluster.sh once per batch
# number in BATCH_NUMS. Edit that list to match how many batches
# step2_sample_wills_reproduce_cluster.sh created (its NUM_BATCH).
#
# Usage: bash commands/step3_collect_wills_reproduce_batches_cluster.sh
# =============================================================================

set -e
set -o pipefail

# ── PBS Configuration ─────────────────────────────────────────────────────────
PBS_NCPUS=96

# ── Python / Environment ──────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="decoder_confidence"

# ── Simulation Parameters ─────────────────────────────────────────────────────
# Edit this list to match NUM_BATCH from step2_sample_wills_reproduce_cluster.sh.
BATCH_NUMS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

# ── Internal Setup ────────────────────────────────────────────────────────────
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COLLECT_SCRIPT="${ROOT_DIR}/commands/step3_collect_wills_reproduce_cluster.sh"

for BATCH_NUM in "${BATCH_NUMS[@]}"; do
    echo "Submitting wills_reproduce collect job for batch ${BATCH_NUM}..."
    BATCH_NUM="${BATCH_NUM}" \
    PBS_JOB_NAME="wills-${BATCH_NUM}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${COLLECT_SCRIPT}"
done
