from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import stim

from decoder_confidence.config import SamplingConfig


@dataclass(frozen=True)
class OutputLayout:
	root_dir: Path
	dem_path: Path
	metadata_path: Path
	sampled_data_dir: Path


def resolve_output_layout(
	out_dir: Path,
	circuit_id: str,
	xyz_decoding: bool,
) -> OutputLayout:
	root_dir = out_dir / f"{circuit_id},xyz={xyz_decoding}"
	return OutputLayout(
		root_dir=root_dir,
		dem_path=root_dir / "dem.dem",
		metadata_path=root_dir / "metadata.json",
		sampled_data_dir=root_dir / "sampled_data",
	)

def detect_data_qubits(circuit: stim.Circuit) -> list[int]:
    """Detect data qubits as those that are only measured once in a circuit.

    Warning: This is hacky and likely will only work with your typical memory circuits.
    """
    qubit_times_measured = [0 for qubit in range(circuit.num_qubits)]

    for inst in circuit:
        if inst.name.startswith("M") and not inst.gate_args_copy():
            for qubit in inst.targets_copy():
                qubit_times_measured[qubit.qubit_value] += 1

    return [
        qubit
        for qubit, times_measured in enumerate(qubit_times_measured)
        if times_measured == 1
    ]


def filter_detectors_by_basis(
    circuit: stim.Circuit,
    basis: str,
    qubits: list[int] | None = None,
) -> stim.Circuit | tuple[stim.Circuit, list[str]]:
    """Return a new circuit filtering any detectors which do not detect the specified basis for the input qubits.

    Args:
        circuit: The original circuit
        basis: "X" or "Z"
        qubits: Data qubits to inject test errors on. Should typically be data qubits. Defaults
            to automatically detected data qubits which may not be robust.

    returns:
        The filtered circuit
    """
    assert basis in ("X", "Z")

    pauli_error = "Z" if basis == "X" else "X"

    circuit = circuit.flattened()

    noiseless_circuit = circuit.without_noise()
    sampler = noiseless_circuit.compile_detector_sampler()
    reference_detectors, reference_observables = sampler.sample(
        1, separate_observables=True
    )
    reference_detectors = reference_detectors[0, :]
    reference_observables = reference_observables[0, :]
    num_detectors = len(reference_detectors)

    detector_is_sensitive = np.full(num_detectors, False, dtype=bool)

    if qubits is None:
        to_test = detect_data_qubits(noiseless_circuit)
    else:
        to_test = qubits

    to_test_set = set(to_test)

    inst_idx = 0
    while to_test:
        for qubit in to_test:
            injected_circuit = stim.Circuit()
            injected_circuit += noiseless_circuit
            injected_circuit.insert(
                inst_idx,
                stim.CircuitInstruction(f"{pauli_error}_ERROR", [qubit], [1.0]),
            )

            injected_sampler = injected_circuit.compile_detector_sampler()
            injected_detectors, injected_observables = injected_sampler.sample(
                1, separate_observables=True
            )
            injected_detectors = injected_detectors[0, :]
            injected_observables = injected_observables[0, :]

            detectors_flipped = np.where(reference_detectors != injected_detectors)
            detector_is_sensitive[detectors_flipped] = True

        to_test = []
        for inst in noiseless_circuit[inst_idx:]:
            # Is a reset we must inject errors after
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(to_test_set)
                break

    filtered_circuit = stim.Circuit()
    detector_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            to_insert = detector_is_sensitive[detector_idx]
            detector_idx += 1
            if not to_insert:
                continue
        filtered_circuit.append(inst)
    return filtered_circuit

def _parse_circuit_stem(stem: str) -> dict[str, Any] | None:
	parts = stem.split(",")
	values: dict[str, str] = {}
	for part in parts:
		if "=" not in part:
			continue
		key, value = part.split("=", 1)
		values[key] = value

	if "code" not in values or "d" not in values or "noisemodel" not in values or "p" not in values:
		return None
	rounds_value = values.get("rounds") or values.get("r")
	if rounds_value is None:
		return None

	try:
		d_value = int(values["d"])
	except ValueError:
		return None
	try:
		p_value = float(values["p"])
	except ValueError:
		return None

	return {
		"code": values["code"],
		"d": d_value,
		"rounds": rounds_value,
		"noisemodel": values["noisemodel"],
		"p": p_value,
		"use_both": values.get("use_both"),  # "True"/"False" for BB code, None otherwise
	}


def find_circuit_file(circuits_dir: Path, config: SamplingConfig) -> Path:
	if not circuits_dir.exists():
		raise FileNotFoundError(f"circuits directory not found: {circuits_dir}")

	is_bb_code = "bivariate_bicycle" in config.code

	matches: list[Path] = []
	for path in circuits_dir.rglob("*.stim"):
		parsed = _parse_circuit_stem(path.stem)
		if parsed is None:
			continue
		if parsed["code"] != config.code:
			continue
		if parsed["d"] != config.d:
			continue
		if parsed["rounds"] != config.rounds:
			continue
		if parsed["noisemodel"] != config.noise_model:
			continue
		if not math.isclose(parsed["p"], config.p, rel_tol=0.0, abs_tol=1e-12):
			continue
		# For BB code, match use_both against xyz_decoding flag.
		# use_both=True  ↔ xyz_decoding=True  (all stabilizer bases included)
		# use_both=False ↔ xyz_decoding=False (single-basis circuit pre-filtered)
		if is_bb_code:
			expected_use_both = "True" if config.xyz_decoding else "False"
			if parsed.get("use_both") != expected_use_both:
				continue
		matches.append(path)

	if not matches:
		expected = (
			f"code={config.code},d={config.d},rounds={config.rounds},"
			f"noisemodel={config.noise_model},p={config.p}"
		)
		if is_bb_code:
			expected += f",use_both={'True' if config.xyz_decoding else 'False'}"
		expected += ".stim"
		raise FileNotFoundError(
			"Circuit not found under circuits/. Expected a file named like: "
			f"{expected}"
		)
	if len(matches) > 1:
		match_list = "\n".join(str(path) for path in matches)
		raise FileExistsError(
			"Multiple circuit files matched the requested parameters:\n" + match_list
		)

	return matches[0]


def load_circuit(circuit_path: Path) -> stim.Circuit:
	return stim.Circuit.from_file(str(circuit_path))


def generate_dem(circuit: stim.Circuit, dem_path: Path, *, decompose_errors: bool = False) -> bool:
	if dem_path.exists():
		return False

	dem_path.parent.mkdir(parents=True, exist_ok=True)
	dem = circuit.detector_error_model(decompose_errors=decompose_errors)
	dem.to_file(str(dem_path))
	return True


def build_metadata(
	config: SamplingConfig,
	circuit_path: Path,
	layout: OutputLayout,
	batch_sizes: list[int],
) -> dict[str, Any]:
	return {
		"schema_version": 1,
		"code": config.code,
		"d": config.d,
		"rounds": config.rounds,
		"noise_model": config.noise_model,
		"p": config.p,
		"num_shots": config.num_shots,
		"num_batch": config.num_batch,
		"batch_sizes": batch_sizes,
		"det_sample_seed": config.det_sample_seed,
		"xyz_decoding": config.xyz_decoding,
		"circuit_id": circuit_path.stem,
		"circuit_path": str(circuit_path),
		"output_dir": str(layout.root_dir),
		"dem_path": str(layout.dem_path),
		"sampled_data_dir": str(layout.sampled_data_dir),
		"sample_format": "b8",
		"append_observables": True,
		"stim_version": getattr(stim, "__version__", "unknown"),
	}


def write_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	with metadata_path.open("w", encoding="utf-8") as handle:
		json.dump(metadata, handle, indent=2, sort_keys=True, ensure_ascii=True)
		handle.write("\n")
