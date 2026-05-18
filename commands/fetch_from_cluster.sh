#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="cluster"

# List remote paths to include in the download (multiple entries allowed).
# - Directories are included as whole directories.
# - Globs (*, ?, []) include all matching entries at that level.
#   Example: "/path/to/results/*" includes everything directly under results.
REMOTE_SOURCES=(
	"/home/u592165f/research/decoder-confidence/cluster/sim_data/code=surface_code_Z,d=5,rounds=1,noisemodel=bitflip,p=0.1,xyz=True"
    "/home/u592165f/research/decoder-confidence/cluster/sim_data/code=surface_code_Z,d=7,rounds=1,noisemodel=bitflip,p=0.1,xyz=True"
    "/home/u592165f/research/decoder-confidence/cluster/sim_data/code=surface_code_Z,d=9,rounds=1,noisemodel=bitflip,p=0.1,xyz=True"
)

# Final destination on the local machine
LOCAL_PATH="/home/quantum_teresheys/workspace/color_code_project/concatenated-decoder/cluster_outputs/"

# Temporary directory on Windows (e.g. C:\temp\cluster_outputs)
WINDOWS_TEMP_PATH="/mnt/c/temp/cluster_outputs"

# Ensure required local directories exist
mkdir -p "$LOCAL_PATH"
mkdir -p "$WINDOWS_TEMP_PATH"

SESSION_DIR_NAME="download_$(date +%Y%m%d_%H%M%S)"
WINDOWS_SESSION_PATH="$WINDOWS_TEMP_PATH/$SESSION_DIR_NAME"
LOCAL_TEMP_DIR="$LOCAL_PATH/.tmp_$SESSION_DIR_NAME"
LOCAL_ARCHIVE_NAME="$SESSION_DIR_NAME.tar.gz"
LOCAL_ARCHIVE_PATH="$LOCAL_TEMP_DIR/$LOCAL_ARCHIVE_NAME"

mkdir -p "$WINDOWS_SESSION_PATH"
mkdir -p "$LOCAL_TEMP_DIR"

move_downloads_into_local() {
	local src_dir="$1"
	local dest_dir="$2"
	shopt -s dotglob nullglob
	local items=("$src_dir"/*)
	shopt -u dotglob nullglob
	if (( ${#items[@]} == 0 )); then
		echo "No files downloaded to: $src_dir" >&2
		return 1
	fi
	mv "${items[@]}" "$dest_dir/"
}

create_remote_archive() {
	local session_dir_name="$1"
	local archive_path="$2"

	ssh "$REMOTE_HOST" bash -s -- "$session_dir_name" "$archive_path" "${REMOTE_SOURCES[@]}" <<'REMOTE_EOF'
set -euo pipefail

session_dir_name="$1"
archive_path="$2"
shift 2
remote_sources=("$@")

tar_args=()
shopt -s nullglob
for src in "${remote_sources[@]}"; do
	if [[ "$src" == *"*"* || "$src" == *"?"* || "$src" == *"["* ]]; then
		matches=($src)
		if (( ${#matches[@]} == 0 )); then
			echo "No matches for pattern: $src" >&2
			exit 1
		fi
		for match in "${matches[@]}"; do
			tar_args+=( -C "$(dirname "$match")" "$(basename "$match")" )
		done
	else
		if [[ ! -e "$src" ]]; then
			echo "Missing path: $src" >&2
			exit 1
		fi
		tar_args+=( -C "$(dirname "$src")" "$(basename "$src")" )
	fi
done

if (( ${#tar_args[@]} == 0 )); then
	echo "No files to archive." >&2
	exit 1
fi

tar -czf "$archive_path" --transform "s#^#${session_dir_name}/#" "${tar_args[@]}"
REMOTE_EOF
}

cleanup_remote_archive() {
	local remote_tmp_dir="$1"
	local archive_path="$2"
	ssh "$REMOTE_HOST" "rm -f '$archive_path' && rmdir '$remote_tmp_dir'"
}

extract_local_archive() {
	local archive_path="$1"
	tar -xzf "$archive_path" -C "$LOCAL_PATH"
	rm -f "$archive_path"
	rmdir "$LOCAL_TEMP_DIR" 2>/dev/null || true
}

REMOTE_TMP_DIR=$(ssh "$REMOTE_HOST" "mktemp -d -t dc_fetch_XXXXXX")
REMOTE_ARCHIVE_PATH="$REMOTE_TMP_DIR/$LOCAL_ARCHIVE_NAME"

echo "Creating remote archive: $REMOTE_HOST:$REMOTE_ARCHIVE_PATH"
create_remote_archive "$SESSION_DIR_NAME" "$REMOTE_ARCHIVE_PATH"

if command -v scp.exe >/dev/null 2>&1; then
	# Convert to Windows path format
	WIN_SESSION_DEST_PATH=$(wslpath -w "$WINDOWS_SESSION_PATH")

	# Use Windows scp.exe to download via Windows networking
	echo "Downloading archive (via scp.exe): $REMOTE_HOST:$REMOTE_ARCHIVE_PATH"
	scp.exe "$REMOTE_HOST:$REMOTE_ARCHIVE_PATH" "$WIN_SESSION_DEST_PATH\\"

	# Move from Windows storage to WSL, then extract locally
	move_downloads_into_local "$WINDOWS_SESSION_PATH" "$LOCAL_TEMP_DIR"
else
	# Non-WSL (pure Linux, etc.): download directly into the local temp dir
	echo "Downloading archive: $REMOTE_HOST:$REMOTE_ARCHIVE_PATH"
	scp "$REMOTE_HOST:$REMOTE_ARCHIVE_PATH" "$LOCAL_TEMP_DIR/"
fi

cleanup_remote_archive "$REMOTE_TMP_DIR" "$REMOTE_ARCHIVE_PATH"
extract_local_archive "$LOCAL_ARCHIVE_PATH"

echo "Transfer completed."
echo "Saved to: $LOCAL_PATH"