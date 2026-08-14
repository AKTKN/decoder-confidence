#!/usr/bin/env bash
# =============================================================================
# PBS Job Submitter — wills_reproduce: Step 2 (create DEM + sample)
#
# Generates the DEM and detector samples needed to reproduce Figure 2(a) of
# Wills, Yoder & Chuang, "Forced Gap Post-Selection for Quantum LDPC Codes
# and their Operations" (arXiv:2605.20346): idling [[72, 12, 6]] bivariate
# bicycle code, 6 rounds, p = 1e-3. The corresponding .stim circuit already
# exists at:
#   circuits/bivariate_bicycle_code_Z/code=bivariate_bicycle_code_Z,d=6,rounds=6,noisemodel=uniform,p=0.001,use_both=False,ibm_reproduce=True.stim
#
# Output is written to a data directory dedicated to this reproduction
# (kept separate from cluster/simulation_result, which backs the main
# confidence-metric pipeline), so re-running this does not touch or
# collide with existing results.
#
# Usage: bash commands/step2_sample_wills_reproduce_cluster.sh
#
# Note on shot count: Figure 2(a) resolves logical error rate down to
# roughly 3.5e-8/round at its ~1% post-selection endpoint. Reaching that
# tail with reasonable statistics needs order 1e8 total shots -- far more
# than the NUM_SHOTS default below, which is only a starting point to
# validate the pipeline end-to-end. Scale NUM_SHOTS/NUM_BATCH up according
# to your cluster budget (each wills_reproduce decode is much more
# expensive per shot than a plain decode -- see step3's script for details
# -- so plan step3 wall-time accordingly before committing to a large
# shot count here).
# =============================================================================

set -e
set -o pipefail

# ── PBS Configuration ─────────────────────────────────────────────────────────
# PBS_WALLTIME="04:00:00"
PBS_NCPUS=1
# PBS_MEM="8gb"

# ── Python / Environment ──────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="decoder_confidence"
# MODULE_LOAD="python/3.10"

# ── Simulation Parameters ─────────────────────────────────────────────────────
CODE="bivariate_bicycle_code_Z"
OUT_DIR="cluster/wills_reproduce_result"
NOISE_MODEL="uniform"
ROUNDS=6
D=6
P=0.001
XYZ_DECODING=False
NUM_SHOTS="${NUM_SHOTS:-2000000}"
NUM_BATCH="${NUM_BATCH:-20}"
DET_SAMPLE_SEED="${DET_SAMPLE_SEED:-0}"
SAMPLING_METHOD=unified

# ── Internal Setup ────────────────────────────────────────────────────────────
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="${ROOT_DIR}/cluster/logs"
mkdir -p "$LOG_DIR"

PBS_JOB_NAME="${PBS_JOB_NAME:-wills-sample}"

JOB_SCRIPT=$(mktemp /tmp/dc_wills_sample_XXXXXX.sh)
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
PBSSCRIPT

JOB_ID=$(qsub "$JOB_SCRIPT")
echo "Submitted job: $JOB_ID"
echo "Logs: ${LOG_DIR}/${PBS_JOB_NAME}.o / .e"
