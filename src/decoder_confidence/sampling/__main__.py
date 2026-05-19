from __future__ import annotations

import logging
from pathlib import Path

from decoder_confidence.config import compute_batch_sizes, parse_args
from decoder_confidence.sampling.dem import (
    build_metadata,
    find_circuit_file,
    generate_dem,
    load_circuit,
    resolve_output_layout,
    write_metadata,
)
from decoder_confidence.sampling.sampler import sample_batches


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dem_decompose_errors(code: str, xyz_decoding: bool) -> bool:
    """Return True if the DEM should be generated with decompose_errors=True.

    surface_code with xyz_decoding=False: use decompose_errors=True so that
    all errors are represented as graph edges (≤2-detector components).
    This lets the decoder work with edge matrices and avoids the 0-detector
    logical-error artefact introduced by basis filtering.
    """
    if code.startswith("surface_code") and not xyz_decoding:
        return True
    # bivariate_bicycle_code: circuit is already pre-filtered when xyz_decoding=False
    #   (use_both=False in filename), so standard DEM generation is correct.
    # superdense_color_code: not yet implemented (see comment below).
    return False


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        config = parse_args(argv)
        circuits_dir = _repo_root() / "circuits"
        circuit_path = find_circuit_file(circuits_dir, config)
        circuit = load_circuit(circuit_path)

        # --- Basis filtering has been removed entirely. ---
        # Previously filter_detectors_by_basis was applied here for xyz_decoding=False.
        # This was found to create 0-detector logical-error artefacts for si1000 and
        # other noise models with correlated X/Z errors.  Each code now handles its
        # own detector structure:
        #
        #   surface_code (xyz_decoding=False):
        #       Use the full circuit, generate DEM with decompose_errors=True.
        #       Step-3 decoding uses edge matrices (edge_check_matrix etc.).
        #
        #   bivariate_bicycle_code (xyz_decoding=False):
        #       The circuit file already has use_both=False in its name; it was
        #       pre-built with only the relevant stabilizer basis.  Sample as-is.
        #
        #   superdense_color_code: NOT YET IMPLEMENTED.
        #       # TODO: implement superdense_color_code sampling support.

        layout = resolve_output_layout(
            Path(config.out_dir), circuit_path.stem, config.xyz_decoding
        )
        layout.root_dir.mkdir(parents=True, exist_ok=True)
        layout.sampled_data_dir.mkdir(parents=True, exist_ok=True)

        decompose = _dem_decompose_errors(config.code, config.xyz_decoding)
        generate_dem(circuit, layout.dem_path, decompose_errors=decompose)
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
