from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from decoder_confidence.config import DecodingResult

if TYPE_CHECKING:
    import stim


class DecoderBase(ABC):
    @abstractmethod
    def decode(self, syndromes: np.ndarray) -> DecodingResult:
        ...


DecoderFactory = Callable[..., DecoderBase]


class SharedEnvDecoderFactory(ABC):
    """Decoder factory that requires a single shared environment across threads.

    Implement this when decoder sessions are limited (e.g. Gurobi WLS).
    The execution manager will call ``build_env()`` once, then
    ``build_decoder(env, dem)`` once per worker thread.
    """

    @abstractmethod
    def build_env(self) -> Any:
        """Create and start the shared environment. Called exactly once."""
        ...

    @abstractmethod
    def build_decoder(self, env: Any, dem: stim.DetectorErrorModel) -> DecoderBase:
        """Create a decoder that uses the shared env. Called once per thread."""
        ...

    def __call__(self, dem: stim.DetectorErrorModel | None = None) -> DecoderBase:
        """Single-shot fallback so this also satisfies DecoderFactory."""
        if dem is None:
            raise ValueError("dem is required for SharedEnvDecoderFactory")
        return self.build_decoder(self.build_env(), dem)


@dataclass(frozen=True)
class DetsFileInfo:
    path: Path
    total_shots: int
    shot_id_offset: int


@dataclass(frozen=True)
class SimulationTask:
    dets_path: Path
    start_shot_index: int
    num_shots: int
    batch_id: int
    shot_id_offset: int


@dataclass(frozen=True)
class WorkerResult:
    status: str
    duration_s: float
    output_path: Path
    num_shots: int
    batch_id: int
    message: str | None = None


@dataclass(frozen=True)
class WorkerConfig:
    dem_path: Path
    output_dir: Path
    decoder_factory: DecoderFactory
    core_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ExecutionConfig:
    dem_path: Path
    sampled_data_dir: Path
    output_dir: Path
    decoder_factory: DecoderFactory
    num_workers: int
    initial_probe_shots: int = 20
    target_task_seconds: float = 30.0
    max_shots_per_task: int | None = None
    core_ids: tuple[int, ...] | None = None
    dets_paths: tuple[Path, ...] | None = None
    verbose: bool = False
    max_chunk_files: int | None = 1000
    merge_chunk_group_size: int | None = None
