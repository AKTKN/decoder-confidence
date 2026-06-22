#!/usr/bin/env bash
# =============================================================================
# PBS Job Submitter — Forcing Degradation: Collect (Batch Submission)
# Usage: bash commands/forcing_degradation_collect_batches_cluster.sh
# =============================================================================

set -e
set -o pipefail

# ── PBS Configuration ─────────────────────────────────────────────────────────
# PBS_WALLTIME="04:00:00"
PBS_NCPUS=96
# PBS_MEM="8gb"

# ── Python / Environment ──────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="decoder_confidence"

# ── Simulation Parameters ─────────────────────────────────────────────────────
# Edit this list to submit the same forcing-degradation job for multiple batches.
BATCH_NUMS=(1 2)

# ── Internal Setup ────────────────────────────────────────────────────────────
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FORCING_SCRIPT="${ROOT_DIR}/commands/forcing_degradation_collect_cluster.sh"

for BATCH_NUM in "${BATCH_NUMS[@]}"; do
    echo "Submitting forcing-degradation job for batch ${BATCH_NUM}..."
    BATCH_NUM="${BATCH_NUM}" \
    PBS_JOB_NAME="FDG-${BATCH_NUM}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${FORCING_SCRIPT}"
done
