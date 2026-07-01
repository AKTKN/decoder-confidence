from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Sequence

from decoder_confidence.execution.models import IncompleteRange

INCOMPLETE_SHOTS_FILENAME = "incomplete_shots.json"


def write_incomplete_shots(path: Path, incomplete: Sequence[IncompleteRange]) -> None:
    ranges = [
        {
            "shot_id_start": r.shot_id_start,
            "shot_id_end": r.shot_id_end,
            "num_shots": r.shot_id_end - r.shot_id_start,
            "batch_id": r.batch_id,
            "dets_path": r.dets_path,
            "reason": r.reason,
            "message": r.message,
            "attempts": r.attempts,
        }
        for r in incomplete
    ]
    payload = {
        "schema_version": 1,
        "ranges": ranges,
        "total_incomplete_shots": sum(r["num_shots"] for r in ranges),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{path.name}"
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def read_incomplete_shots(path: Path) -> list[IncompleteRange]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        IncompleteRange(
            shot_id_start=r["shot_id_start"],
            shot_id_end=r["shot_id_end"],
            batch_id=r["batch_id"],
            dets_path=r["dets_path"],
            reason=r["reason"],
            message=r.get("message"),
            attempts=r.get("attempts", 1),
        )
        for r in payload["ranges"]
    ]
