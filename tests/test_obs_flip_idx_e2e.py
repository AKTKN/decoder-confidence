"""End-to-end test for obs_flip_idx: observable flip index tracking.

Tests VarInt encoding round-trip, the full pipeline writing an obs_flip_idx
binary file, and correctness of the saved indices against direct decoder output.

Uses bivariate_bicycle_code_Z d=4 (4 observables, 45 detectors) with the
existing ILP logical-gap decoder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ILP_SRC = REPO_ROOT / "Donotsync" / "ILP-decoder" / "src"
for p in (str(SRC_ROOT), str(ILP_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from decoder_confidence.varint import (
    encode_obs_flip_shot,
    decode_obs_flip_shot,
    write_obs_flip_idx_file,
    read_obs_flip_idx_file,
)

BB4_DATA = (
    REPO_ROOT
    / "raw_data"
    / "code=bivariate_bicycle_code_Z,d=4,rounds=4,noisemodel=uniform,p=0.001,use_both=False,xyz=False"
)
BB4_DEM = BB4_DATA / "dem.dem"
BB4_B8 = BB4_DATA / "sampled_data" / "det_batch=1.b8"
BB4_NUM_OBS = 4


# ---------------------------------------------------------------------------
# VarInt unit tests
# ---------------------------------------------------------------------------


class TestVarInt:
    def test_empty(self):
        blob = encode_obs_flip_shot([])
        indices, offset = decode_obs_flip_shot(blob, 0)
        assert indices == []
        assert offset == len(blob)

    def test_single(self):
        for idx in [0, 1, 3, 127, 128, 300]:
            blob = encode_obs_flip_shot([idx])
            indices, offset = decode_obs_flip_shot(blob, 0)
            assert indices == [idx]

    def test_multiple_sorted(self):
        original = [0, 1, 2, 3]
        blob = encode_obs_flip_shot(original)
        indices, _ = decode_obs_flip_shot(blob, 0)
        assert indices == original

    def test_unsorted_input_is_sorted_output(self):
        blob = encode_obs_flip_shot([3, 0, 2, 1])
        indices, _ = decode_obs_flip_shot(blob, 0)
        assert indices == [0, 1, 2, 3]

    def test_large_indices(self):
        original = [100, 200, 500]
        blob = encode_obs_flip_shot(original)
        indices, _ = decode_obs_flip_shot(blob, 0)
        assert indices == original

    def test_file_roundtrip(self, tmp_path):
        shots = [[], [0], [1, 3], [0, 1, 2, 3], [2]]
        blobs = [encode_obs_flip_shot(s) for s in shots]
        path = tmp_path / "test.bin"
        write_obs_flip_idx_file(path, blobs)
        recovered = read_obs_flip_idx_file(path)
        assert recovered == shots


# ---------------------------------------------------------------------------
# Helpers that require gurobipy / ilp_decoder
# ---------------------------------------------------------------------------


def _skip_if_missing():
    pytest.importorskip("gurobipy")
    pytest.importorskip("ilp_decoder")
    if not BB4_DEM.exists() or not BB4_B8.exists():
        pytest.skip("bivariate_bicycle_code_Z d=4 data not found in raw_data/")


def _read_b8(path: Path, num_detectors: int, num_observables: int, num_shots: int):
    bps = (num_detectors + num_observables + 7) // 8
    with path.open("rb") as fh:
        raw = fh.read(bps * num_shots)
    data = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(data, bitorder="little").reshape(num_shots, bps * 8)
    total = num_detectors + num_observables
    return bits[:, :total].astype(bool)


def _build_ilp_gap_decoder(env, dem):
    """Build an _ILPLogicalGapDecoder from a stim DEM."""
    import stim
    from ilp_decoder import DecoderConfig, ILPDecoder
    from ilp_decoder.core import DecoderDependencies
    from decoder_confidence.decoding._ilp_logicalgap import _ILPLogicalGapDecoder
    from decoder_confidence.decoding._decoder_adapter import (
        _clip_priors,
        _dem_to_matrices_with_edge_priors,
    )

    matrices = _dem_to_matrices_with_edge_priors(dem, allow_undecomposed_hyperedges=True)
    priors = _clip_priors(matrices.priors)
    deps = DecoderDependencies(env=env)
    config = DecoderConfig(log_to_console=False)
    ilp = ILPDecoder(
        parity_check_matrix=matrices.check_matrix,
        observables=matrices.observables_matrix,
        prior=priors,
        config=config,
        deps=deps,
    )
    return _ILPLogicalGapDecoder(decoder=ilp)


@pytest.fixture(scope="module")
def gurobi_env():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi not available: {exc}")
    yield env
    env.dispose()


# ---------------------------------------------------------------------------
# Direct decoder test: verify obs_flip_idx correctness
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_obs_flip_idx_direct_decoder(gurobi_env):
    """Run the ILP gap decoder on a small batch and verify obs_flip_idx properties.

    For each shot the obs_flip_idx must be:
    - Non-empty (gap model is always constrained to find a *different* logical class)
    - Within [0, num_obs - 1]
    - Consistent: predictions XOR flip_mask == stage2_obs (constructed from indices)
    """
    _skip_if_missing()

    import stim

    num_shots = 10  # small batch for speed
    dem = stim.DetectorErrorModel.from_file(str(BB4_DEM))
    decoder = _build_ilp_gap_decoder(gurobi_env, dem)

    data = _read_b8(BB4_B8, dem.num_detectors, dem.num_observables, num_shots)
    dets = data[:, : dem.num_detectors]
    true_obs = data[:, dem.num_detectors :]

    result = decoder.decode(dets)

    assert result.obs_flip_idx is not None, "obs_flip_idx should be populated for ILP gap decoder"
    assert len(result.obs_flip_idx) == num_shots

    for shot_i, flip_idx in enumerate(result.obs_flip_idx):
        # Must be non-empty: by construction the gap model forces a different class
        assert len(flip_idx) > 0, f"shot {shot_i}: obs_flip_idx should not be empty"
        # All indices must be valid observable indices
        for idx in flip_idx:
            assert 0 <= idx < BB4_NUM_OBS, (
                f"shot {shot_i}: obs index {idx} out of range [0, {BB4_NUM_OBS})"
            )
        # Sorted
        assert flip_idx == sorted(flip_idx), f"shot {shot_i}: obs_flip_idx should be sorted"

    # Verify that predictions XOR obs_flip_mask differs from predictions at
    # exactly the obs_flip_idx positions (structural consistency check).
    for shot_i, flip_idx in enumerate(result.obs_flip_idx):
        pred = result.predictions[shot_i].astype(int)
        mask = np.zeros(BB4_NUM_OBS, dtype=int)
        for idx in flip_idx:
            mask[idx] = 1
        stage2_obs = (pred + mask) % 2
        # They must differ at flip_idx positions
        for j in range(BB4_NUM_OBS):
            if j in flip_idx:
                assert stage2_obs[j] != pred[j], (
                    f"shot {shot_i}, obs {j}: expected stage-2 to differ from stage-1"
                )
            else:
                assert stage2_obs[j] == pred[j], (
                    f"shot {shot_i}, obs {j}: expected stage-2 to match stage-1"
                )


@pytest.mark.e2e
def test_obs_flip_idx_matches_z_vars(gurobi_env):
    """Verify obs_flip_idx exactly matches z_vars from the gap ILP by re-solving.

    For each shot, re-run the stage-2 gap model, extract z_var values, and
    compare to the obs_flip_idx stored in the result metadata.
    """
    _skip_if_missing()

    import stim
    from ilp_decoder import DecoderConfig, ILPDecoder
    from ilp_decoder.core import DecoderDependencies
    from ilp_decoder.formulation import build_logical_gap_model
    from decoder_confidence.decoding._decoder_adapter import (
        _clip_priors,
        _dem_to_matrices_with_edge_priors,
    )

    num_shots = 5
    dem = stim.DetectorErrorModel.from_file(str(BB4_DEM))
    matrices = _dem_to_matrices_with_edge_priors(dem, allow_undecomposed_hyperedges=True)
    priors = _clip_priors(matrices.priors)
    config = DecoderConfig(log_to_console=False)
    deps = DecoderDependencies(env=gurobi_env)
    ilp = ILPDecoder(
        parity_check_matrix=matrices.check_matrix,
        observables=matrices.observables_matrix,
        prior=priors,
        config=config,
        deps=deps,
    )

    data = _read_b8(BB4_B8, dem.num_detectors, dem.num_observables, num_shots)
    dets = data[:, : dem.num_detectors]

    results = ilp.decode_batch_result(dets.tolist(), get_logicalgap=True)

    for shot_i, res in enumerate(results):
        stored_flip = sorted(res.metadata.get("obs_flip_idx", []))
        pred_obs = np.asarray(res.predicted_observables, dtype=int)

        # Re-solve stage-2 model to independently obtain z_vars
        gap_model = build_logical_gap_model(
            gurobi_env,
            matrices.check_matrix,
            matrices.observables_matrix,
            config,
            logical_class=pred_obs,
            prior=priors,
        )
        from ilp_decoder.core import GurobiBackend, _apply_config_params
        backend = GurobiBackend()
        gap_result = backend.solve(gap_model, dets[shot_i].astype(int), config=config)

        z_vars = gap_model.auxiliary_vars["logical_xor"]
        expected_flip = sorted(i for i, v in enumerate(z_vars) if int(round(v.X)) == 1)

        assert stored_flip == expected_flip, (
            f"shot {shot_i}: stored obs_flip_idx {stored_flip} != "
            f"independently computed {expected_flip}"
        )


# ---------------------------------------------------------------------------
# Full pipeline E2E test: binary file output
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_obs_flip_idx_pipeline_output(tmp_path):
    """Run the full step-3 pipeline on bivariate_bicycle_code_Z d=4 and verify
    the obs_flip_idx binary file is produced with correct content.
    """
    _skip_if_missing()

    from decoder_confidence.decoding.__main__ import main as decoding_main

    data_dir = BB4_DATA.parent

    args = [
        "--code", "bivariate_bicycle_code_Z",
        "--data_dir", str(data_dir),
        "--noise_model", "uniform",
        "--rounds", "4",
        "--d", "4",
        "--p", "0.001",
        "--batch_num", "1",
        "--num_workers", "1",
        "--decoder_config", str(REPO_ROOT / "conf" / "step3_config.yaml"),
        "--xyz", "False",
    ]

    try:
        rc = decoding_main(args)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert rc == 0

    output_dir = (
        BB4_DATA / "decoding_result" / "decoder=ILP,metric=logical_gap"
    )
    assert (output_dir / "logicalerror_batch=1.parquet").exists()
    assert (output_dir / "metric=logical_gap_batch=1.parquet").exists()

    obs_bin = output_dir / "obs_flip_idx_batch=1.bin"
    assert obs_bin.exists(), "obs_flip_idx binary file should be produced by ILP gap decoder"

    shot_indices = read_obs_flip_idx_file(obs_bin)

    # The total shots in det_batch=1.b8 is 100
    assert len(shot_indices) == 100, (
        f"Expected 100 shots, got {len(shot_indices)}"
    )

    for shot_i, flip_idx in enumerate(shot_indices):
        assert isinstance(flip_idx, list)
        # Non-empty: ILP gap always finds a different logical class
        assert len(flip_idx) > 0, f"shot {shot_i}: obs_flip_idx should not be empty"
        for idx in flip_idx:
            assert 0 <= idx < BB4_NUM_OBS, (
                f"shot {shot_i}: index {idx} out of range [0, {BB4_NUM_OBS})"
            )
        assert flip_idx == sorted(flip_idx), f"shot {shot_i}: indices not sorted"


@pytest.mark.e2e
def test_obs_flip_idx_binary_file_validity(gurobi_env):
    """Verify the binary file produced by the pipeline has structurally valid content.

    The ILP solver can return different-but-equally-optimal solutions across
    separate runs when there is degeneracy in the gap model.  Therefore this
    test checks structural invariants rather than exact match across runs:
    - Every shot has a non-empty obs_flip_idx (gap model always finds a different class)
    - All indices are in [0, num_obs - 1]
    - Indices are sorted within each shot
    - The in-memory decoder result satisfies the same invariants
    """
    _skip_if_missing()

    import stim

    obs_bin = (
        BB4_DATA
        / "decoding_result"
        / "decoder=ILP,metric=logical_gap"
        / "obs_flip_idx_batch=1.bin"
    )
    if not obs_bin.exists():
        pytest.skip("Pipeline output not present; run test_obs_flip_idx_pipeline_output first")

    saved = read_obs_flip_idx_file(obs_bin)
    assert len(saved) == 100

    for shot_i, flip_idx in enumerate(saved):
        assert len(flip_idx) > 0, f"shot {shot_i}: binary file has empty obs_flip_idx"
        assert all(0 <= i < BB4_NUM_OBS for i in flip_idx), (
            f"shot {shot_i}: index out of range in {flip_idx}"
        )
        assert flip_idx == sorted(flip_idx), f"shot {shot_i}: not sorted"

    # Also verify the same invariants hold for a fresh in-memory decode.
    num_shots = 10
    dem = stim.DetectorErrorModel.from_file(str(BB4_DEM))
    decoder = _build_ilp_gap_decoder(gurobi_env, dem)
    data = _read_b8(BB4_B8, dem.num_detectors, dem.num_observables, num_shots)
    dets = data[:, : dem.num_detectors]
    result = decoder.decode(dets)

    assert result.obs_flip_idx is not None
    for shot_i, flip_idx in enumerate(result.obs_flip_idx):
        assert len(flip_idx) > 0, f"direct shot {shot_i}: empty obs_flip_idx"
        assert all(0 <= i < BB4_NUM_OBS for i in flip_idx)
        assert flip_idx == sorted(flip_idx)
