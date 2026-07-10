#!/usr/bin/env bash
#
# tar_extract.sh
#
# Purpose:
#   Extract a .tar archive into a specified output directory, preserving
#   the same directory structure that existed at the time of compression.
#
# Usage:
#   Edit the INPUT_TAR and OUTPUT_DIR variables below, then run:
#     ./tar_extract.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Edit these two paths before running the script.
# ---------------------------------------------------------------------------
INPUT_TAR="/home/quantum_teresheys/workspace/decoder-confidence/simulation_result/backup_20260709_094314.tar"
OUTPUT_DIR="/home/quantum_teresheys/workspace/decoder-confidence/simulation_result"
# ---------------------------------------------------------------------------

if [[ ! -f "$INPUT_TAR" ]]; then
  echo "Error: input file '$INPUT_TAR' does not exist." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# tar_backup.sh stores members with their original absolute paths. GNU tar
# strips the leading "/" from each member name on extraction, so the
# original directory structure is recreated as a relative tree under
# OUTPUT_DIR (for example, /home/user/project_a becomes
# OUTPUT_DIR/home/user/project_a).
tar -xvf "$INPUT_TAR" -C "$OUTPUT_DIR"

echo "Extraction complete. Files restored under: $OUTPUT_DIR"