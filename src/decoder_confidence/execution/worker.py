from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import polars as pl
import stim

from decoder_confidence.varint import encode_obs_flip_shot
from decoder_confidence.execution.hashing import chunk_path
from decoder_confidence.execution.models import (
    DecoderFactory,
    SimulationTask,
    WorkerConfig,
    WorkerResult,
)


@dataclass
class WorkerState:
    dem: stim.DetectorErrorModel
    num_detectors: int
    num_observables: int
    bytes_per_shot: int
    decoder: Any
    decoder_name: str
    output_dir: Path
    decoder_accepts_true_obs: bool = False


_STATE: WorkerState | None = None
_INIT_ERROR: str | None = None


def _apply_affinity(core_ids: tuple[int, ...] | None) -> None:
    if not core_ids:
        return
    try:
        identity = mp.current_process()._identity
        worker_index = identity[0] - 1 if identity else 0
        if 0 <= worker_index < len(core_ids):
            os.sched_setaffinity(0, {core_ids[worker_index]})
    except (AttributeError, OSError):
        return


def _get_dem_counts(dem: stim.DetectorErrorModel) -> tuple[int, int]:
    try:
        return int(dem.num_detectors), int(dem.num_observables)
    except AttributeError as exc:
        raise ValueError(
            "stim.DetectorErrorModel missing num_detectors/num_observables"
        ) from exc


def _build_decoder(factory: DecoderFactory, dem: stim.DetectorErrorModel) -> Any:
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory()
    if len(params) == 1:
        return factory(dem)
    return factory()


def init_worker(config: WorkerConfig) -> None:
    global _STATE, _INIT_ERROR

    _STATE = None
    _INIT_ERROR = None

    try:
        _apply_affinity(config.core_ids)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        dem = stim.DetectorErrorModel.from_file(str(config.dem_path))
        num_detectors, num_observables = _get_dem_counts(dem)
        bytes_per_shot = (num_detectors + num_observables + 7) // 8

        decoder = _build_decoder(config.decoder_factory, dem)
        decoder_name = decoder.__class__.__name__

        try:
            decoder_accepts_true_obs = (
                "true_obs" in inspect.signature(decoder.decode).parameters
            )
        except (TypeError, ValueError):
            decoder_accepts_true_obs = False

        _STATE = WorkerState(
            dem=dem,
            num_detectors=num_detectors,
            num_observables=num_observables,
            bytes_per_shot=bytes_per_shot,
            decoder=decoder,
            decoder_name=decoder_name,
            output_dir=config.output_dir,
            decoder_accepts_true_obs=decoder_accepts_true_obs,
        )
    except Exception:
        _INIT_ERROR = traceback.format_exc()


def read_b8_slice(
    dets_path: Path,
    start_shot_index: int,
    num_shots: int,
    *,
    bytes_per_shot: int,
    num_detectors: int,
    num_observables: int,
) -> np.ndarray:
    if num_shots <= 0:
        raise ValueError(f"num_shots must be > 0 but got {num_shots}")

    offset = bytes_per_shot * start_shot_index
    bytes_to_read = bytes_per_shot * num_shots

    with dets_path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(bytes_to_read)

    if len(raw) != bytes_to_read:
        raise ValueError(
            f"Expected {bytes_to_read} bytes but read {len(raw)} from {dets_path}"
        )

    data = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(data, bitorder="little")
    bits = bits.reshape(num_shots, bytes_per_shot * 8)
    total_bits = num_detectors + num_observables
    return bits[:, :total_bits].astype(np.bool_, copy=False)


def _normalize_predictions(
    predictions: Any, num_shots: int, num_observables: int
) -> np.ndarray:
    if num_observables == 0:
        return np.zeros((num_shots, 0), dtype=bool)

    arr = np.asarray(predictions)
    if arr.ndim == 0:
        if num_observables != 1:
            raise ValueError(
                "Scalar predictions are only valid for a single observable."
            )
        arr = np.full((num_shots, 1), bool(arr))
    elif arr.ndim == 1:
        if num_observables != 1:
            raise ValueError(
                "1D predictions are only valid for a single observable."
            )
        if arr.shape[0] != num_shots:
            raise ValueError(
                f"predictions length {arr.shape[0]} != num_shots {num_shots}"
            )
        arr = arr.reshape(num_shots, 1)
    elif arr.ndim == 2:
        if arr.shape != (num_shots, num_observables):
            raise ValueError(
                "predictions must be shaped (num_shots, num_observables)"
            )
    else:
        raise ValueError("predictions must be a scalar, 1D, or 2D array")

    return arr.astype(np.bool_, copy=False)


def _normalize_metrics(metrics: dict[str, Any], num_shots: int) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for key, value in metrics.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = np.full(num_shots, arr, dtype=arr.dtype)
        elif arr.ndim == 1:
            if arr.shape[0] != num_shots:
                raise ValueError(
                    f"metric {key} length {arr.shape[0]} != num_shots {num_shots}"
                )
        elif arr.ndim == 2 and arr.shape[0] == num_shots and arr.shape[1] == 1:
            arr = arr.reshape(num_shots)
        else:
            raise ValueError(
                f"metric {key} must be scalar or 1D array of length num_shots"
            )
        normalized[f"metric_{key}"] = arr
    return normalized


def run_task(task: SimulationTask) -> WorkerResult:
    if _STATE is None:
        message = _INIT_ERROR or "Worker not initialized. Call init_worker first."
        return WorkerResult(
            status="error",
            duration_s=0.0,
            output_path=Path(),
            num_shots=task.num_shots,
            batch_id=task.batch_id,
            message=message,
        )

    output_path = chunk_path(_STATE.output_dir, task, _STATE.decoder_name)
    if output_path.exists():
        return WorkerResult(
            status="skipped",
            duration_s=0.0,
            output_path=output_path,
            num_shots=task.num_shots,
            batch_id=task.batch_id,
        )

    start_time = time.perf_counter()
    try:
        data = read_b8_slice(
            task.dets_path,
            task.start_shot_index,
            task.num_shots,
            bytes_per_shot=_STATE.bytes_per_shot,
            num_detectors=_STATE.num_detectors,
            num_observables=_STATE.num_observables,
        )

        dets = data[:, : _STATE.num_detectors]
        obs = data[:, _STATE.num_detectors :]

        if _STATE.decoder_accepts_true_obs and _STATE.num_observables > 0:
            result = _STATE.decoder.decode(dets, true_obs=obs)
        else:
            result = _STATE.decoder.decode(dets)
        predictions = _normalize_predictions(
            result.predictions, task.num_shots, _STATE.num_observables
        )

        if _STATE.num_observables > 0:
            mismatch = predictions ^ obs
            is_logical_error = mismatch.any(axis=1)
        else:
            is_logical_error = np.zeros(task.num_shots, dtype=bool)
        logical_error_override = result.metrics.get("__is_logical_error")
        if logical_error_override is not None:
            override = np.asarray(logical_error_override, dtype=np.bool_)
            if override.shape != (task.num_shots,):
                raise ValueError(
                    "__is_logical_error must be a 1D boolean array of length "
                    f"{task.num_shots}, got shape {override.shape}"
                )
            is_logical_error = is_logical_error | override

        shot_ids = task.shot_id_offset + task.start_shot_index + np.arange(
            task.num_shots, dtype=np.int64
        )

        columns: dict[str, Any] = {
            "shot_id": shot_ids,
            "is_logical_error": is_logical_error,
        }
        columns.update(_normalize_metrics(result.metrics, task.num_shots))

        if result.obs_flip_idx is not None:
            blobs = [encode_obs_flip_shot(idxs) for idxs in result.obs_flip_idx]
            columns["obs_flip_idx"] = pl.Series("obs_flip_idx", blobs, dtype=pl.Binary)

        df = pl.DataFrame(columns)
        df.write_parquet(output_path, compression="zstd")

        status = "ok"
        message = None
    except Exception as exc:
        status = "error"
        message = str(exc)
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

    duration = time.perf_counter() - start_time
    return WorkerResult(
        status=status,
        duration_s=duration,
        output_path=output_path,
        num_shots=task.num_shots,
        batch_id=task.batch_id,
        message=message,
    )
