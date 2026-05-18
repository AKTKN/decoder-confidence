from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from decoder_confidence.config import DecodingResult


class DecoderBase(ABC):
    @abstractmethod
    def decode(self, syndromes: np.ndarray) -> DecodingResult:
        ...


DecoderFactory = Callable[..., DecoderBase]


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
