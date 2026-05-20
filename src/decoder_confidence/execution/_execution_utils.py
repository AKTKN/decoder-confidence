"""Pure utility functions shared by multiprocessing and threaded execution managers."""
from __future__ import annotations

import os
import resource
import sys
from pathlib import Path
from typing import Iterable

import polars as pl
import stim

from decoder_confidence.execution.models import DetsFileInfo


def get_dem_counts(dem_path: Path) -> tuple[int, int]:
    dem = stim.DetectorErrorModel.from_file(str(dem_path))
    try:
        return int(dem.num_detectors), int(dem.num_observables)
    except AttributeError as exc:
        raise ValueError(
            "stim.DetectorErrorModel missing num_detectors/num_observables"
        ) from exc


def bytes_per_shot_count(num_detectors: int, num_observables: int) -> int:
    return (num_detectors + num_observables + 7) // 8


def discover_dets_files(sampled_data_dir: Path) -> list[Path]:
    return sorted(sampled_data_dir.glob("*.b8"))


def build_file_infos(dets_paths: Iterable[Path], bps: int) -> list[DetsFileInfo]:
    infos: list[DetsFileInfo] = []
    offset = 0
    for path in dets_paths:
        size = path.stat().st_size
        if size % bps != 0:
            raise ValueError(
                f"File size for {path} not aligned to bytes_per_shot={bps}"
            )
        total_shots = size // bps
        infos.append(DetsFileInfo(path=path, total_shots=total_shots, shot_id_offset=offset))
        offset += total_shots
    return infos


def select_core_ids(
    num_workers: int, core_ids: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    if core_ids:
        return core_ids
    try:
        available = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None
    if not available:
        return None
    return tuple(available[:num_workers])


def rss_mb() -> float:
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage_kb / (1024.0 * 1024.0)
    return usage_kb / 1024.0


def resolve_merge_group_size(
    max_chunk_files: int | None, merge_chunk_group_size: int | None
) -> int | None:
    if max_chunk_files is None or max_chunk_files <= 0:
        return None
    if merge_chunk_group_size is None or merge_chunk_group_size <= 0:
        return min(200, max_chunk_files)
    return merge_chunk_group_size


def next_part_index(part_dir: Path) -> int:
    if not part_dir.exists():
        return 1
    max_index = 0
    for path in part_dir.glob("part_*.parquet"):
        stem = path.stem
        if "_" not in stem:
            continue
        suffix = stem.split("_", 1)[1]
        try:
            max_index = max(max_index, int(suffix))
        except ValueError:
            continue
    return max_index + 1


def merge_chunks_into_part(
    chunk_paths: list[Path], part_dir: Path, part_index: int
) -> Path:
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"part_{part_index:06d}.parquet"
    scan = pl.scan_parquet([str(path) for path in chunk_paths])
    scan.sink_parquet(part_path, compression="zstd")
    for path in chunk_paths:
        try:
            path.unlink()
        except OSError:
            pass
    return part_path
