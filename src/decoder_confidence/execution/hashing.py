from __future__ import annotations

import hashlib
from pathlib import Path

from decoder_confidence.execution.models import SimulationTask


def stable_task_id(task: SimulationTask, decoder_name: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(decoder_name.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(str(task.dets_path).encode("utf-8"))
    hasher.update(b"|")
    hasher.update(
        f"{task.start_shot_index}:{task.num_shots}:{task.batch_id}:{task.shot_id_offset}".encode(
            "utf-8"
        )
    )
    return hasher.hexdigest()


def chunk_filename(task: SimulationTask, decoder_name: str) -> str:
    task_id = stable_task_id(task, decoder_name)
    return f"chunk_{task.batch_id:06d}_{task_id[:8]}.parquet"


def chunk_path(output_dir: Path, task: SimulationTask, decoder_name: str) -> Path:
    return output_dir / chunk_filename(task, decoder_name)
