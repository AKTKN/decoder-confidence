#!/usr/bin/env bash
# =============================================================================
# PBS Job Submitter — wills_reproduce: Step 3 (collect, single batch)
#
# Runs the wills_reproduce metric (Wills, Yoder & Chuang, "Forced Gap
# Post-Selection for Quantum LDPC Codes and their Operations",
# arXiv:2605.20346, Figure 2(a) reproduction) over one sampled batch,
# using the same `decoder_confidence.decoding` entry point and
# batch/sample-loading conventions as commands/step3_collect.sh -- the
# only differences are the metric-specific decoder_config and the
# dedicated wills_reproduce data directory (kept separate from
# cluster/simulation_result so this reproduction cannot collide with the
# main confidence-metric pipeline's results).
#
# Usage: bash commands/step3_collect_wills_reproduce_cluster.sh
#   BATCH_NUM=<n> bash commands/step3_collect_wills_reproduce_cluster.sh   # override batch
#
# Run step2_sample_wills_reproduce_cluster.sh first to create the DEM and
# sampled_data batches this reads from.
#
# Cost note: each shot here costs one baseline Relay-BP call with up to
# num_sets=1201 legs, plus up to 12 forced Relay-BP calls with up to
# num_sets=25 legs each (Appendix A settings, conf/step3_config_wills_reproduce.yaml)
# -- substantially more Relay-BP legs per shot than the production
# forced_gap_ml run (num_sets=300 shared, stop_nconv=1) that averaged
# ~27s/shot. Benchmark on a small NUM_WORKERS/short batch before sizing
# PBS_WALLTIME for a full run.
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

# ── Simulation Parameters ─────────────────────────────────────────────────────
CODE="bivariate_bicycle_code_Z"
DATA_DIR="cluster/wills_reproduce_result"
NOISE_MODEL="uniform"
ROUNDS=6
D=6
P=0.001
XYZ=False
BATCH_NUM="${BATCH_NUM:-1}"
NUM_WORKERS=96
DECODER_CONFIG="cluster/conf/step3_config_wills_reproduce.yaml"
VERBOSE=False
IBM_REPRODUCE=true

# ── Internal Setup ────────────────────────────────────────────────────────────
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="${ROOT_DIR}/cluster/logs"
mkdir -p "$LOG_DIR"

PBS_JOB_NAME="${PBS_JOB_NAME:-wills-${BATCH_NUM}}"

IBM_REPRODUCE_FLAG=""
if [ "${IBM_REPRODUCE}" = "true" ]; then
    IBM_REPRODUCE_FLAG="--ibm_reproduce"
fi

# NUM_WORKERS must match PBS_NCPUS
if [ "$NUM_WORKERS" -ne "$PBS_NCPUS" ]; then
    echo "Warning: NUM_WORKERS (${NUM_WORKERS}) != PBS_NCPUS (${PBS_NCPUS})." \
         "Consider keeping them in sync." >&2
fi

JOB_SCRIPT=$(mktemp /tmp/dc_wills_collect_XXXXXX.sh)
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

"${PYTHON_BIN}" -m decoder_confidence.decoding \
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
echo "Logs: ${LOG_DIR}/${PBS_JOB_NAME}.o / .e"
