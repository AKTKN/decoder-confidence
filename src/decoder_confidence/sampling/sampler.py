from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import stim

from decoder_confidence.config import validate_seed


def _batch_offsets(batch_sizes: list[int]) -> list[tuple[int, int]]:
	"""Return half-open shot ranges for each batch in the one-shot sample."""
	offsets: list[tuple[int, int]] = []
	start = 0
	for shots in batch_sizes:
		end = start + int(shots)
		offsets.append((start, end))
		start = end
	return offsets


def sample_batches(
	circuit: stim.Circuit,
	sampled_data_dir: Path,
	batch_sizes: list[int],
	det_sample_seed: int,
	*,
	append_observables: bool = True,
	sample_format: str = "b8",
) -> list[Path]:
	if sample_format != "b8":
		raise ValueError(f"sample_batches only supports 'b8', got {sample_format!r}")

	sampled_data_dir.mkdir(parents=True, exist_ok=True)
	outputs: list[Path] = []
	total_shots = int(sum(batch_sizes))
	if total_shots <= 0:
		return outputs

	validate_seed(det_sample_seed)

	# 2026-06-25: The previous implementation compiled one sampler per batch
	# with seed = det_sample_seed + batch_index - 1.  That made the sampled
	# shot set depend on num_batch even when num_shots and det_sample_seed were
	# unchanged.  The current implementation samples the full experiment once
	# with det_sample_seed, then writes deterministic slices to batch files.
	sampler = circuit.compile_detector_sampler(seed=det_sample_seed)
	data = sampler.sample(total_shots, append_observables=append_observables)

	for batch_index, (start, end) in enumerate(_batch_offsets(batch_sizes), start=1):
		if end <= start:
			continue
		out_path = sampled_data_dir / f"det_batch={batch_index}.{sample_format}"
		if out_path.exists():
			logging.info("Skipping existing batch file: %s", out_path)
			continue

		_write_b8(out_path, data[start:end])
		outputs.append(out_path)

	return outputs


def _write_b8(path: Path, data: np.ndarray) -> None:
	"""Write a boolean (num_shots, num_bits) array to a file in stim b8 format.

	Stim's b8 format stores each bit as the LSB of each successive byte position
	(little-endian bit order).  Rows are padded with zero bits to the next byte
	boundary before packing.
	"""
	num_shots, num_bits = data.shape
	bytes_per_shot = (num_bits + 7) // 8
	pad = bytes_per_shot * 8 - num_bits
	if pad:
		padding = np.zeros((num_shots, pad), dtype=np.uint8)
		data_u8 = np.concatenate([data.astype(np.uint8), padding], axis=1)
	else:
		data_u8 = data.astype(np.uint8)
	packed = np.packbits(data_u8, axis=1, bitorder="little")
	path.write_bytes(packed.tobytes())


def sample_batches_from_dem(
	dem: stim.DetectorErrorModel,
	sampled_data_dir: Path,
	batch_sizes: list[int],
	det_sample_seed: int,
	*,
	sample_format: str = "b8",
) -> list[Path]:
	"""Sample directly from a DetectorErrorModel and write per-batch b8 files.

	Used when xyz_decoding=False for surface_code / superdense_color_code: the
	DEM has already been filtered to the relevant stabiliser basis, so sampling
	from it directly produces syndrome data that is consistent with the DEM
	without any additional column selection.

	The written data contains the detector bits followed by the observable bits,
	matching the layout expected by the decoder workers (same as sample_batches
	with append_observables=True).

	Args:
		dem: Filtered DetectorErrorModel.
		sampled_data_dir: Directory in which to write per-batch b8 files.
		batch_sizes: Number of shots per batch.
		det_sample_seed: Base random seed for the full one-shot sample.
		sample_format: Only "b8" is supported.

	Returns:
		List of paths to the written batch files.
	"""
	if sample_format != "b8":
		raise ValueError(f"sample_batches_from_dem only supports 'b8', got {sample_format!r}")

	sampled_data_dir.mkdir(parents=True, exist_ok=True)
	outputs: list[Path] = []
	total_shots = int(sum(batch_sizes))
	if total_shots <= 0:
		return outputs

	validate_seed(det_sample_seed)

	# 2026-06-25: The old batch-local sampling scheme used seeds
	# det_sample_seed, det_sample_seed + 1, ... for separate batch samplers.
	# With the same total shots, changing num_batch therefore changed the
	# sampled population.  We now sample total_shots once and split the resulting
	# array, matching "sample once, then partition" semantics.
	sampler = dem.compile_sampler(seed=det_sample_seed)
	# Returns (dets, obs, errors); errors is None by default.
	dets, obs, _ = sampler.sample(total_shots)

	# Concatenate detector bits and observable bits, then write as b8.
	data = np.concatenate([dets, obs], axis=1)
	for batch_index, (start, end) in enumerate(_batch_offsets(batch_sizes), start=1):
		if end <= start:
			continue
		out_path = sampled_data_dir / f"det_batch={batch_index}.{sample_format}"
		if out_path.exists():
			logging.info("Skipping existing batch file: %s", out_path)
			continue
		_write_b8(out_path, data[start:end])
		outputs.append(out_path)

	return outputs
