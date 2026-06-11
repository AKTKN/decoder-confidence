"""Unit tests for decoding.incomplete (incomplete_shots.json manifest)."""
from __future__ import annotations

import json

from decoder_confidence.decoding.incomplete import (
    INCOMPLETE_SHOTS_FILENAME,
    read_incomplete_shots,
    write_incomplete_shots,
)
from decoder_confidence.execution.models import IncompleteRange


def _sample_ranges() -> list[IncompleteRange]:
    return [
        IncompleteRange(
            shot_id_start=480000,
            shot_id_end=480512,
            batch_id=938,
            dets_path="/data/det_batch=10.b8",
            reason="timeout",
            message="exceeded 312.4s (attempt 2)",
            attempts=2,
        ),
        IncompleteRange(
            shot_id_start=12000,
            shot_id_end=12064,
            batch_id=42,
            dets_path="/data/det_batch=3.b8",
            reason="error",
            message="Logical gap solve failed or objective unavailable.",
            attempts=2,
        ),
    ]


def test_round_trip(tmp_path):
    ranges = _sample_ranges()
    path = tmp_path / INCOMPLETE_SHOTS_FILENAME

    write_incomplete_shots(path, ranges)
    recovered = read_incomplete_shots(path)

    assert recovered == ranges


def test_written_payload_schema(tmp_path):
    ranges = _sample_ranges()
    path = tmp_path / INCOMPLETE_SHOTS_FILENAME
    write_incomplete_shots(path, ranges)

    payload = json.loads(path.read_text())

    assert payload["schema_version"] == 1
    assert payload["total_incomplete_shots"] == 512 + 64

    by_batch = {r["batch_id"]: r for r in payload["ranges"]}
    assert by_batch[938]["num_shots"] == 512
    assert by_batch[938]["reason"] == "timeout"
    assert by_batch[42]["num_shots"] == 64
    assert by_batch[42]["reason"] == "error"


def test_manifest_lifecycle_write_then_clear_on_success(tmp_path):
    """Mirrors the __main__.py/resume.py pattern: write when non-empty,
    delete a stale manifest from a prior run when the new run is fully
    successful (incomplete == []).
    """
    path = tmp_path / INCOMPLETE_SHOTS_FILENAME

    # First run: some shots incomplete -> manifest written.
    write_incomplete_shots(path, _sample_ranges())
    assert path.exists()

    # Second run: everything succeeded -> manifest is cleared.
    incomplete: list[IncompleteRange] = []
    if incomplete:
        write_incomplete_shots(path, incomplete)
    elif path.exists():
        path.unlink()

    assert not path.exists()
