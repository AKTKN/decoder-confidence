from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoder_confidence.decoding.__main__ import main as decoding_main
from decoder_confidence.sampling.__main__ import main as sampling_main

CIRCUIT_STEM = "code=surface_code_Z,d=5,rounds=5,noisemodel=si1000,p=0.001"


def _run_sampling(out_dir: Path, xyz_decoding: bool) -> Path:
    args = [
        "--code",
        "surface_code_Z",
        "--out_dir",
        str(out_dir),
        "--noise_model",
        "si1000",
        "--rounds",
        "5",
        "--d",
        "5",
        "--p",
        "0.001",
        "--num_shots",
        "2",
        "--det_sample_seed",
        "0",
        "--num_batch",
        "1",
        "--xyz_decoding",
        str(xyz_decoding),
    ]
    rc = sampling_main(args)
    assert rc == 0
    return out_dir / f"{CIRCUIT_STEM},xyz={xyz_decoding}"


@pytest.fixture()
def sampled_output(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    return _run_sampling(out_dir, xyz_decoding=False)


@pytest.mark.e2e
def test_step2_sampling_xyz_suffix(sampled_output: Path) -> None:
    assert sampled_output.exists()
    assert (sampled_output / "dem.dem").exists()
    assert (sampled_output / "sampled_data" / "det_batch=1.b8").exists()


@pytest.mark.e2e
def test_step3_decoding_e2e(sampled_output: Path) -> None:
    pytest.importorskip("gurobipy")
    pytest.importorskip("ilp_decoder")
    pytest.importorskip("beliefmatching")

    data_dir = sampled_output.parent
    args = [
        "--code",
        "surface_code_Z",
        "--data_dir",
        str(data_dir),
        "--noise_model",
        "si1000",
        "--rounds",
        "5",
        "--d",
        "5",
        "--p",
        "0.001",
        "--batch_num",
        "1",
        "--num_workers",
        "1",
        "--decoder_config",
        str(REPO_ROOT / "conf" / "step3_config.yaml"),
        "--xyz",
        "False",
    ]
    try:
        rc = decoding_main(args)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert rc == 0

    output_dir = (
        sampled_output
        / "decoding_result"
        / "decoder=ILP,metric=logical_gap"
    )
    assert (output_dir / "logicalerror_batch=1.parquet").exists()
    assert (output_dir / "metric=logical_gap_batch=1.parquet").exists()
    assert (output_dir / "metadata.json").exists()
