from __future__ import annotations

import logging
from pathlib import Path

from decoder_confidence.config import compute_batch_sizes, parse_args
from decoder_confidence.sampling.dem import (
    build_metadata,
    filter_detectors_by_basis,
    find_circuit_file,
    generate_dem,
    load_circuit,
    resolve_output_layout,
    write_metadata,
)
from decoder_confidence.sampling.sampler import sample_batches


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        config = parse_args(argv)
        circuits_dir = _repo_root() / "circuits"
        circuit_path = find_circuit_file(circuits_dir, config)
        circuit = load_circuit(circuit_path)
        if not config.xyz_decoding:
            circuit = filter_detectors_by_basis(circuit, basis="Z")

        layout = resolve_output_layout(
            Path(config.out_dir), circuit_path.stem, config.xyz_decoding
        )
        layout.root_dir.mkdir(parents=True, exist_ok=True)
        layout.sampled_data_dir.mkdir(parents=True, exist_ok=True)

        generate_dem(circuit, layout.dem_path)
        batch_sizes = compute_batch_sizes(config.num_shots, config.num_batch)
        metadata = build_metadata(config, circuit_path, layout, batch_sizes)
        write_metadata(layout.metadata_path, metadata)

        sample_batches(
            circuit,
            layout.sampled_data_dir,
            batch_sizes,
            config.det_sample_seed,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        logging.error(str(exc))
        return 2
    except Exception as exc:  # pragma: no cover - last resort for unexpected failures
        logging.exception("Unexpected error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
