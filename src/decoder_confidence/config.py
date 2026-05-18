from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from pathlib import Path

UINT64_MAX = 2**64 - 1

@dataclass(frozen=True)
class SamplingConfig:
	code: str
	out_dir: Path
	noise_model: str
	rounds: str
	d: int
	p: float
	num_shots: int
	det_sample_seed: int
	num_batch: int
	xyz_decoding: bool

@dataclass
class DecodingResult:
	"""Standard output for all decoders."""
	predictions: np.ndarray
	metrics: Dict[str, Any] = field(default_factory=dict)


def validate_seed(seed: int) -> None:
	if seed < 0 or seed > UINT64_MAX:
		raise ValueError(f"det_sample_seed must be in [0, {UINT64_MAX}] but got {seed}")


def compute_batch_sizes(num_shots: int, num_batch: int) -> list[int]:
	if num_shots < 1:
		raise ValueError(f"num_shots must be >= 1 but got {num_shots}")
	if num_batch < 1:
		raise ValueError(f"num_batch must be >= 1 but got {num_batch}")
	if num_batch > num_shots:
		raise ValueError(
			"num_batch must be <= num_shots to avoid empty batches "
			f"(num_batch={num_batch}, num_shots={num_shots})"
		)

	base = num_shots // num_batch
	remainder = num_shots % num_batch
	return [base + 1 if i < remainder else base for i in range(num_batch)]


def _parse_bool(value: str | bool) -> bool:
	if isinstance(value, bool):
		return value
	value_str = str(value).strip().lower()
	if value_str in {"1", "true", "t", "yes", "y"}:
		return True
	if value_str in {"0", "false", "f", "no", "n"}:
		return False
	raise ValueError(f"xyz_decoding must be a boolean value but got {value}")


def parse_args(argv: list[str] | None = None) -> SamplingConfig:
	parser = argparse.ArgumentParser(
		description="Generate a DEM and sample detector data from a Stim circuit."
	)
	parser.add_argument("--code", required=True, help="Code name used in circuit filename")
	parser.add_argument("--out_dir", required=True, help="Output directory for DEM and samples")
	parser.add_argument("--noise_model", required=True, help="Noise model name in circuit filename")
	parser.add_argument("--rounds", required=True, help="Rounds token used in circuit filename")
	parser.add_argument("--d", required=True, type=int, help="Code distance")
	parser.add_argument("--p", required=True, type=float, help="Noise strength")
	parser.add_argument("--num_shots", required=True, type=int, help="Total number of shots")
	parser.add_argument(
		"--det_sample_seed",
		required=True,
		type=int,
		help="Base seed for detector sampling",
	)
	parser.add_argument("--num_batch", required=True, type=int, help="Number of batches")
	parser.add_argument(
		"--xyz_decoding",
		required=False,
		default=False,
		type=_parse_bool,
		help="If true, keep detectors for both X and Z basis",
	)

	args = parser.parse_args(argv)

	rounds = str(args.rounds).strip()
	if not rounds:
		raise ValueError("rounds must be a non-empty string")
	if args.d < 1:
		raise ValueError(f"d must be >= 1 but got {args.d}")
	validate_seed(args.det_sample_seed)
	compute_batch_sizes(args.num_shots, args.num_batch)

	return SamplingConfig(
		code=args.code,
		out_dir=Path(args.out_dir),
		noise_model=args.noise_model,
		rounds=rounds,
		d=args.d,
		p=args.p,
		num_shots=args.num_shots,
		det_sample_seed=args.det_sample_seed,
		num_batch=args.num_batch,
		xyz_decoding=bool(args.xyz_decoding),
	)
