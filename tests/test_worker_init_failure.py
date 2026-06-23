from __future__ import annotations

from pathlib import Path

import pytest
import stim

from decoder_confidence.execution import worker
from decoder_confidence.execution.models import SimulationTask, WorkerConfig


def test_worker_initializer_failure_is_reported_as_task_error(tmp_path: Path) -> None:
    dem_path = tmp_path / "empty.dem"
    dem_path.write_text(str(stim.DetectorErrorModel()), encoding="utf-8")

    def broken_factory(_dem):
        raise TypeError("decoder constructor failed")

    try:
        worker.init_worker(
            WorkerConfig(
                dem_path=dem_path,
                output_dir=tmp_path / "chunks",
                decoder_factory=broken_factory,
            )
        )
        result = worker.run_task(
            SimulationTask(
                dets_path=tmp_path / "det_batch=1.b8",
                start_shot_index=0,
                num_shots=1,
                batch_id=1,
                shot_id_offset=0,
            )
        )
    finally:
        worker._STATE = None
        worker._INIT_ERROR = None

    assert result.status == "error"
    assert result.output_path == Path()
    assert "decoder constructor failed" in (result.message or "")
    assert "Traceback" in (result.message or "")
