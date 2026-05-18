from __future__ import annotations

import logging
from pathlib import Path

import stim

from decoder_confidence.config import UINT64_MAX, validate_seed


def sample_batches(
	circuit: stim.Circuit,
	sampled_data_dir: Path,
	batch_sizes: list[int],
	det_sample_seed: int,
	*,
	append_observables: bool = True,
	sample_format: str = "b8",
) -> list[Path]:
	sampled_data_dir.mkdir(parents=True, exist_ok=True)
	outputs: list[Path] = []

	for batch_index, shots in enumerate(batch_sizes, start=1):
		if shots <= 0:
			continue
		out_path = sampled_data_dir / f"det_batch={batch_index}.{sample_format}"
		if out_path.exists():
			logging.info("Skipping existing batch file: %s", out_path)
			continue

		seed = det_sample_seed + (batch_index - 1)
		if seed > UINT64_MAX:
			raise ValueError(
				f"det_sample_seed + batch_index must be <= {UINT64_MAX} but got {seed}"
			)
		validate_seed(seed)

		sampler = circuit.compile_detector_sampler(seed=seed)
		sampler.sample_write(
			shots=shots,
			filepath=out_path,
			format=sample_format,
			append_observables=append_observables,
		)
		outputs.append(out_path)

	return outputs
