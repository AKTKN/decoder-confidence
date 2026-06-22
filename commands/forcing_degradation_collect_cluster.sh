#!/usr/bin/env bash
# =============================================================================
# PBS Job Submitter — Forcing Degradation: Collect
# Usage: bash commands/forcing_degradation_collect_cluster.sh
# =============================================================================

set -e
set -o pipefail

# ── PBS Configuration ─────────────────────────────────────────────────────────
# PBS_WALLTIME="04:00:00"
PBS_NCPUS=96
# PBS_MEM="8gb"

# ── Python / Environment ──────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
# Uncomment and set if using a conda environment:
CONDA_ENV="decoder_confidence"
# Uncomment and set if using environment modules:
# MODULE_LOAD="python/3.10"

# ── Simulation Parameters ─────────────────────────────────────────────────────
CODE="bivariate_bicycle_code_Z"
DATA_DIR="cluster/simulation_data_2"
NOISE_MODEL="uniform"
ROUNDS=6
D=6
P=0.003
XYZ=False
BATCH_NUM="${BATCH_NUM:-9}"
NUM_WORKERS=96
DECODER_CONFIG="cluster/conf/forcing_degradation_bplsd.yaml"
VERBOSE=False
IBM_REPRODUCE=true

# ── Internal Setup ────────────────────────────────────────────────────────────
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="${ROOT_DIR}/cluster/logs"
mkdir -p "$LOG_DIR"

PBS_JOB_NAME="${PBS_JOB_NAME:-fdg-${BATCH_NUM}}"

IBM_REPRODUCE_FLAG=""
if [ "${IBM_REPRODUCE}" = "true" ]; then
    IBM_REPRODUCE_FLAG="--ibm_reproduce"
fi

# NUM_WORKERS must match PBS_NCPUS
if [ "$NUM_WORKERS" -ne "$PBS_NCPUS" ]; then
    echo "Warning: NUM_WORKERS (${NUM_WORKERS}) != PBS_NCPUS (${PBS_NCPUS})." \
         "Consider keeping them in sync." >&2
fi

JOB_SCRIPT=$(mktemp /tmp/dc_forcing_degradation_XXXXXX.sh)
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

"${PYTHON_BIN}" -m decoder_confidence.decoding.forcing_degradation_collect \
    --code ${CODE} \
    --data_dir ${DATA_DIR} \
    --noise_model ${NOISE_MODEL} \
    --rounds ${ROUNDS} \
    --d ${D} \
    --p ${P} \
    --xyz ${XYZ} \
    --batch_num ${BATCH_NUM} \
    --num_workers ${NUM_WORKERS} \
    --decoder_config ${ROOT_DIR}/${DECODER_CONFIG} \
    --verbose ${VERBOSE} \
    ${IBM_REPRODUCE_FLAG}
PBSSCRIPT

JOB_ID=$(qsub "$JOB_SCRIPT")
echo "Submitted job: $JOB_ID"
echo "Logs: ${LOG_DIR}/$JOB_ID.o / .e"
