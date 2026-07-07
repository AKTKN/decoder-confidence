from __future__ import annotations

import logging
from pathlib import Path

import stim

from decoder_confidence.config import compute_batch_sizes, parse_args
from decoder_confidence.sampling.dem import (
    build_metadata,
    filter_dem_by_basis,
    find_circuit_file,
    generate_dem,
    load_circuit,
    resolve_output_layout,
    write_metadata,
)
from decoder_confidence.sampling.sampler import sample_batches_from_dem


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _needs_dem_filter(code: str, xyz_decoding: bool) -> bool:
    """Return True if the DEM should be filtered to a single stabiliser basis.

    Applies to surface_code and superdense_color_code when xyz_decoding=False.
    bivariate_bicycle_code uses pre-built single-basis circuit files (use_both=False)
    so no filtering is needed.
    """
    if xyz_decoding:
        return False
    if code.startswith("surface_code"):
        return True
    # superdense_color_code: filter_dem_by_basis supports it, but the full
    # pipeline is not yet implemented.
    # TODO: enable once superdense_color_code pipeline is complete.
    # if code.startswith("superdense_color_code"):
    #     return True
    return False


def _get_remove_basis(code: str) -> str:
    """Determine which detector basis to remove for a single-basis decoding run."""
    if code.endswith("_Z"):
        return "X"   # Z-memory: keep Z stabiliser detectors, remove X detectors
    if code.endswith("_X"):
        return "Z"   # X-memory: keep X stabiliser detectors, remove Z detectors
    raise ValueError(
        f"Cannot infer remove_basis from code name {code!r}. "
        "Expected code name ending in '_Z' or '_X'."
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        config = parse_args(argv)
        circuits_dir = _repo_root() / "circuits"
        circuit_path = find_circuit_file(circuits_dir, config)
        circuit = load_circuit(circuit_path)

        layout = resolve_output_layout(
            Path(config.out_dir), circuit_path.stem, config.xyz_decoding
        )
        layout.root_dir.mkdir(parents=True, exist_ok=True)
        layout.sampled_data_dir.mkdir(parents=True, exist_ok=True)

        batch_sizes = compute_batch_sizes(config.num_shots, config.num_batch)
        metadata = build_metadata(config, circuit_path, layout, batch_sizes)
        write_metadata(layout.metadata_path, metadata)

        # --- Step 1: build and save the DEM -----------------------------------
        if _needs_dem_filter(config.code, config.xyz_decoding):
            # surface_code (or future superdense_color_code) with xyz_decoding=False:
            # generate a full DEM then filter to the relevant stabiliser basis.
            full_dem = circuit.detector_error_model(decompose_errors=False)
            if not layout.dem_path.exists():
                filtered_dem = filter_dem_by_basis(full_dem, _get_remove_basis(config.code))
                layout.dem_path.parent.mkdir(parents=True, exist_ok=True)
                filtered_dem.to_file(str(layout.dem_path))
        else:
            # All other cases (xyz_decoding=True for any code, or
            # bivariate_bicycle_code xyz_decoding=False with pre-built circuit):
            # generate a standard full DEM.
            #
            # bivariate_bicycle_code with xyz_decoding=False uses a pre-built
            # circuit (use_both=False in filename) that already contains only the
            # relevant basis — no DEM-level filtering needed.
            #
            # superdense_color_code: NOT YET IMPLEMENTED.
            # TODO: implement superdense_color_code sampling support.
            generate_dem(circuit, layout.dem_path)

        # --- Step 2: sample from the saved DEM --------------------------------
        # Unified for all cases: load the DEM (filtered or full) and sample
        # directly from it via CompiledDemSampler.  This ensures the syndrome
        # data is always consistent with the DEM that step-3 decoding will use.
        dem_for_sampling = stim.DetectorErrorModel.from_file(str(layout.dem_path))
        sample_batches_from_dem(
            dem_for_sampling,
            layout.sampled_data_dir,
            batch_sizes,
            config.det_sample_seed,
            sampling_method=config.sampling_method,
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
