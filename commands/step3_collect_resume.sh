#!/usr/bin/env bash
# =============================================================================
# PBS Job Submitter — Step 3: Resume (Resume an interrupted collect run)
#
# Usage: bash commands/step3_collect_resume.sh
#
# Set OUTPUT_DIR to the *exact* decoding_result/decoder=...,metric=.../ directory
# that was interrupted.  All other experiment settings are derived from that
# directory's location on disk (circuit_dir, dem.dem, sampled_data/).
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
# MODULE_LOAD="python/3.10"

# ── Resume Parameters ─────────────────────────────────────────────────────────
# Full path to the interrupted output directory, e.g.:
#   cluster/simulation_data/code=bivariate_bicycle_code_Z,...,p=0.01/decoding_result/decoder=ILP,metric=logical_gap
OUTPUT_DIR="cluster/simulation_data/FIXME/decoding_result/decoder=ILP,metric=logical_gap"

# Batch number to resume (leave empty to auto-detect when only one batch exists)
BATCH_NUM="${BATCH_NUM:-1}"

NUM_WORKERS=96
DECODER_CONFIG="cluster/conf/step3_config.yaml"
VERBOSE=True
CLEANUP_INTERMEDIATE=True

# ── Internal Setup ────────────────────────────────────────────────────────────
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="${ROOT_DIR}/cluster/logs"
mkdir -p "$LOG_DIR"

PBS_JOB_NAME="${PBS_JOB_NAME:-resume-${BATCH_NUM}}"

# NUM_WORKERS must match PBS_NCPUS
if [ "$NUM_WORKERS" -ne "$PBS_NCPUS" ]; then
    echo "Warning: NUM_WORKERS (${NUM_WORKERS}) != PBS_NCPUS (${PBS_NCPUS})." \
         "Consider keeping them in sync." >&2
fi

# Resolve OUTPUT_DIR to absolute path if relative
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}"
fi

JOB_SCRIPT=$(mktemp /tmp/dc_step3_resume_XXXXXX.sh)
trap 'rm -f "$JOB_SCRIPT"' EXIT

cat > "$JOB_SCRIPT" << PBSSCRIPT
#!/bin/bash
#PBS -N ${PBS_JOB_NAME}
#PBS -l select=1:ncpus=${PBS_NCPUS}
#PBS -o ${LOG_DIR}/${PBS_JOB_NAME}.o
#PBS -e ${LOG_DIR}/${PBS_JOB_NAME}.e
set -e
set -o pipefail

ulimit -s unlimited
cd ${ROOT_DIR}

# Activate environment if needed
source activate ${CONDA_ENV}
# module load ${MODULE_LOAD}

export PYTHONPATH="${ROOT_DIR}/src:\${PYTHONPATH}"

"${PYTHON_BIN}" -m decoder_confidence.decoding.resume \
    --output_dir "${OUTPUT_DIR}" \
    --batch_num ${BATCH_NUM} \
    --num_workers ${NUM_WORKERS} \
    --decoder_config ${ROOT_DIR}/${DECODER_CONFIG} \
    --verbose ${VERBOSE} \
    --cleanup_intermediate ${CLEANUP_INTERMEDIATE}
PBSSCRIPT

JOB_ID=$(qsub "$JOB_SCRIPT")
echo "Submitted job: $JOB_ID"
echo "Logs: ${LOG_DIR}/${PBS_JOB_NAME}.o / .e"
