from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import resource
import sys
from typing import Any, Mapping, Sequence

import stim

from decoder_confidence.decoding.decoder_factory import DecoderConfigInfo
from decoder_confidence.execution.models import IncompleteRange, WorkerResult


@dataclass(frozen=True)
class WorkerSummary:
    ok_shots: int
    skipped_shots: int
    error_shots: int
    ok_duration_s: float


def _get_dem_counts(dem: stim.DetectorErrorModel) -> tuple[int, int]:
    try:
        return int(dem.num_detectors), int(dem.num_observables)
    except AttributeError as exc:
        raise ValueError(
            "stim.DetectorErrorModel missing num_detectors/num_observables"
        ) from exc


def _bytes_per_shot(num_detectors: int, num_observables: int) -> int:
    return (num_detectors + num_observables + 7) // 8


def _total_shots(dets_paths: Sequence[Path], bytes_per_shot: int) -> int:
    total = 0
    for path in dets_paths:
        size = path.stat().st_size
        if size % bytes_per_shot != 0:
            raise ValueError(
                f"File size for {path} not aligned to bytes_per_shot={bytes_per_shot}"
            )
        total += size // bytes_per_shot
    return total


def _summarize_results(results: Sequence[WorkerResult]) -> WorkerSummary:
    ok_shots = 0
    skipped_shots = 0
    error_shots = 0
    ok_duration_s = 0.0
    for result in results:
        if result.status == "ok":
            ok_shots += result.num_shots
            ok_duration_s += result.duration_s
        elif result.status == "skipped":
            skipped_shots += result.num_shots
        else:
            error_shots += result.num_shots
    return WorkerSummary(
        ok_shots=ok_shots,
        skipped_shots=skipped_shots,
        error_shots=error_shots,
        ok_duration_s=ok_duration_s,
    )


def _rss_mb(who: int) -> float:
    usage_kb = resource.getrusage(who).ru_maxrss
    if sys.platform == "darwin":
        return usage_kb / (1024.0 * 1024.0)
    return usage_kb / 1024.0


def _peak_rss_mb() -> float:
    return max(_rss_mb(resource.RUSAGE_SELF), _rss_mb(resource.RUSAGE_CHILDREN))


def _check_matrix_shape(
    dem: stim.DetectorErrorModel, decoder_name: str
) -> tuple[int | None, int | None]:
    if decoder_name.upper() != "ILP":
        return None, None
    try:
        from beliefmatching import detector_error_model_to_check_matrices
    except ImportError:
        return None, None

    mats = detector_error_model_to_check_matrices(
        dem, allow_undecomposed_hyperedges=True
    )
    check_matrix = mats.check_matrix
    return int(check_matrix.shape[0]), int(check_matrix.shape[1])


def _incomplete_entries(
    incomplete_ranges: Sequence[IncompleteRange], reason: str
) -> list[dict[str, Any]] | None:
    entries = [
        {
            "shot_id_start": r.shot_id_start,
            "shot_id_end": r.shot_id_end,
            "batch_id": r.batch_id,
            "message": r.message,
        }
        for r in incomplete_ranges
        if r.reason == reason
    ]
    return entries or None


def build_decoding_metadata(
    *,
    decoder_info: DecoderConfigInfo,
    dem_path: Path,
    dets_paths: Sequence[Path],
    num_workers: int,
    start_time: datetime,
    end_time: datetime,
    results: Sequence[WorkerResult],
    metrics_recorded: Sequence[str],
    incomplete_ranges: Sequence[IncompleteRange] = (),
) -> dict[str, Any]:
    dem = stim.DetectorErrorModel.from_file(str(dem_path))
    num_detectors, num_observables = _get_dem_counts(dem)
    bytes_per_shot = _bytes_per_shot(num_detectors, num_observables)
    total_shots = _total_shots(dets_paths, bytes_per_shot)

    summary = _summarize_results(results)
    avg_time_per_shot = (
        summary.ok_duration_s / summary.ok_shots if summary.ok_shots else None
    )

    check_rows, check_cols = _check_matrix_shape(dem, decoder_info.decoder_name)
    random_seed = decoder_info.decoder_options.get("random_seed")
    solver_options = decoder_info.decoder_options.get("solver_options")
    incomplete_shots = sum(
        r.shot_id_end - r.shot_id_start for r in incomplete_ranges
    )

    return {
        "schema_version": 1,
        "decoder_config": {
            "path": str(decoder_info.config_path),
            "decoder": decoder_info.raw_config.get("decoder"),
            "metric": decoder_info.raw_config.get("metric"),
            "decoder_options": decoder_info.raw_config.get("decoder_options", {}),
            "metric_options": decoder_info.raw_config.get("metric_options", {}),
        },
        "decoder": {
            "name": decoder_info.decoder_name,
            "metric": decoder_info.metric_name,
            "metrics_recorded": list(metrics_recorded),
            "random_seed": random_seed,
        },
        "problem_size": {
            "check_matrix_rows": check_rows,
            "check_matrix_cols": check_cols,
            "num_detectors": num_detectors,
            "num_observables": num_observables,
        },
        "data": {
            "dets_paths": [str(path) for path in dets_paths],
            "dem_path": str(dem_path),
            "bytes_per_shot": bytes_per_shot,
            "total_shots": total_shots,
        },
        "execution": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_seconds": (end_time - start_time).total_seconds(),
            "avg_time_per_shot_seconds": avg_time_per_shot,
            "num_workers": num_workers,
            "executed_shots": summary.ok_shots,
            "skipped_shots": summary.skipped_shots,
            "error_shots": summary.error_shots,
            "incomplete_shots": incomplete_shots,
            "total_shots": total_shots,
            "peak_memory_mb": _peak_rss_mb(),
        },
        "algorithm_status": {
            "solver_options": solver_options,
            "timeouts": _incomplete_entries(incomplete_ranges, "timeout"),
            "nonconverged": None,
            "failed": _incomplete_entries(incomplete_ranges, "error"),
        },
    }


def write_metadata(metadata_path: Path, metadata: Mapping[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
