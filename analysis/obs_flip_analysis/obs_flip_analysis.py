"""Observable flip index analysis: load and visualise obs_flip_idx data.

Usage from a notebook in analysis/obs_flip_analysis/ (or anywhere with sys.path set):

    from obs_flip_analysis import ObsFlipConfig, ObsFlipDataManager, ObsFlipAnalyzer

The module reads the binary ``obs_flip_idx_batch=<N>.bin`` files produced by the
decoder-confidence pipeline alongside ``logicalerror_batch=<N>.parquet`` files.
It provides a histogram of how often each logical observable index appears in the
second-stage flip set, with an option to split shots by whether a logical error occurred.
"""
from __future__ import annotations

import math
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Bootstrap: make decoder_confidence importable even when this module is
# imported from a notebook that has not yet added src/ to sys.path.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from decoder_confidence.varint import read_obs_flip_idx_file  # noqa: E402

# ---------------------------------------------------------------------------
# Directory-name parsing helpers (same conventions as analysis/src/data_manager.py)
# ---------------------------------------------------------------------------

_BATCH_BIN_RE = re.compile(r"obs_flip_idx_batch=(\d+)\.bin$")
_BATCH_PARQUET_RE = re.compile(r"_batch=(\d+)\.parquet$")


def _parse_kv(name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in name.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        result[key.strip()] = val.strip()
    return result


def _strip_quotes(s: str) -> str:
    return s.strip("'\"")


def _match_str(raw: str, expected: Any) -> bool:
    clean = _strip_quotes(raw)
    if isinstance(expected, bool):
        return clean.lower() == ("true" if expected else "false")
    if isinstance(expected, int):
        try:
            return int(clean) == expected
        except ValueError:
            return False
    if isinstance(expected, float):
        try:
            return math.isclose(float(clean), expected, rel_tol=1e-9, abs_tol=1e-15)
        except ValueError:
            return False
    return clean == str(expected)


def _is_circuit_dir(name: str) -> bool:
    kv = _parse_kv(name)
    return "code" in kv and "d" in kv and "p" in kv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ObsFlipConfig:
    """Specifies which dataset to load for observable flip index analysis.

    Parameters
    ----------
    root_dir:
        Parent directory that contains circuit-parameter subdirectories
        (e.g. ``raw_data/``).
    code:
        Code name, e.g. ``"bivariate_bicycle_code_Z"`` or ``"surface_code_Z"``.
    noise_model:
        Noise model key in the directory name, e.g. ``"uniform"`` or ``"phenomenological"``.
    d:
        Code distance.
    rounds:
        Number of rounds (int or string).
    p:
        Physical error probability.
    decoder:
        Decoder name as it appears in the directory, e.g. ``"ILP"`` or ``"MWPM"``.
    metric:
        Metric name, e.g. ``"logical_gap"`` or ``"linearize_logicalgap"``.
    batch_indices:
        Restrict to specific batch numbers. ``None`` loads all available batches.
    num_observables:
        Number of logical observables.  When ``None`` (default) it is inferred
        from the data (max index + 1).
    """

    root_dir: Path
    code: str
    noise_model: str
    d: int
    rounds: int | str
    p: float
    decoder: str
    metric: str
    batch_indices: Optional[list[int]] = None
    num_observables: Optional[int] = None

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.rounds = str(self.rounds)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class ObsFlipData:
    """Loaded observable flip index data for one configuration.

    Attributes
    ----------
    flip_indices:
        Per-shot lists of observable indices where the stage-2 solution differs
        from stage-1. Length == number of shots loaded.
    is_logical_error:
        Boolean array (length == number of shots) indicating whether each shot
        is a logical error.
    num_observables:
        Total number of logical observables.  Used as x-axis range in plots.
    shot_ids:
        Shot IDs corresponding to each entry (aligned with flip_indices).
    """

    flip_indices: list[list[int]]
    is_logical_error: np.ndarray
    num_observables: int
    shot_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def __post_init__(self) -> None:
        self.is_logical_error = np.asarray(self.is_logical_error, dtype=bool)

    @property
    def num_shots(self) -> int:
        return len(self.flip_indices)

    @property
    def num_logical_errors(self) -> int:
        return int(self.is_logical_error.sum())


# ---------------------------------------------------------------------------
# Data manager
# ---------------------------------------------------------------------------


class ObsFlipDataManager:
    """Loads obs_flip_idx binary files and logical-error parquets.

    Parameters
    ----------
    root_dir:
        May be supplied here as a convenience.  If ``None``, the root is taken
        from ``config.root_dir`` in :meth:`load`.
    """

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self._root_dir = Path(root_dir) if root_dir is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, config: ObsFlipConfig) -> ObsFlipData:
        """Find and load all matching obs_flip_idx data.

        Raises
        ------
        FileNotFoundError
            If no matching circuit directory, decoder directory, or binary
            files are found.
        """
        root = self._root_dir if self._root_dir is not None else config.root_dir

        circuit_dir = self._find_circuit_dir(root, config)
        decoder_dir = self._find_decoder_dir(circuit_dir, config)

        all_flip: list[list[int]] = []
        all_errors: list[bool] = []
        all_shot_ids: list[int] = []

        batches_loaded = 0
        for bin_path in sorted(decoder_dir.glob("obs_flip_idx_batch=*.bin")):
            m = _BATCH_BIN_RE.search(bin_path.name)
            if m is None:
                continue
            batch_idx = int(m.group(1))
            if config.batch_indices is not None and batch_idx not in config.batch_indices:
                continue

            le_path = decoder_dir / f"logicalerror_batch={batch_idx}.parquet"
            if not le_path.exists():
                raise FileNotFoundError(
                    f"logicalerror_batch={batch_idx}.parquet not found alongside "
                    f"{bin_path}"
                )

            flip, errors, shot_ids = self._load_batch(bin_path, le_path)
            all_flip.extend(flip)
            all_errors.extend(errors)
            all_shot_ids.extend(shot_ids)
            batches_loaded += 1

        if batches_loaded == 0:
            raise FileNotFoundError(
                f"No obs_flip_idx_batch=*.bin files found in {decoder_dir}"
            )

        # Infer num_observables from data if not specified
        if config.num_observables is not None:
            num_obs = config.num_observables
        else:
            num_obs = max(
                (max(idxs) + 1 for idxs in all_flip if idxs),
                default=0,
            )

        return ObsFlipData(
            flip_indices=all_flip,
            is_logical_error=np.array(all_errors, dtype=bool),
            num_observables=num_obs,
            shot_ids=np.array(all_shot_ids, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_circuit_dir(self, root: Path, config: ObsFlipConfig) -> Path:
        if not root.is_dir():
            raise FileNotFoundError(f"root_dir not found: {root}")

        filters: dict[str, Any] = {
            "code": config.code,
            "noisemodel": config.noise_model,
            "d": config.d,
            "rounds": config.rounds,
            "p": config.p,
        }

        matches: list[Path] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or not _is_circuit_dir(entry.name):
                continue
            params = _parse_kv(entry.name)
            if all(_match_str(params.get(k, ""), v) for k, v in filters.items()):
                matches.append(entry)

        if not matches:
            desc = ", ".join(f"{k}={v}" for k, v in filters.items())
            raise FileNotFoundError(
                f"No circuit directory matched ({desc}) under {root}"
            )
        if len(matches) > 1:
            paths = "\n  ".join(str(p) for p in matches)
            raise FileExistsError(
                f"Multiple circuit directories matched:\n  {paths}"
            )
        return matches[0]

    def _find_decoder_dir(self, circuit_dir: Path, config: ObsFlipConfig) -> Path:
        decoding_dir = circuit_dir / "decoding_result"
        if not decoding_dir.is_dir():
            raise FileNotFoundError(
                f"decoding_result/ not found under {circuit_dir}"
            )

        matches: list[Path] = []
        for entry in sorted(decoding_dir.iterdir()):
            if not entry.is_dir():
                continue
            params = _parse_kv(entry.name)
            decoder_ok = _match_str(params.get("decoder", ""), config.decoder)
            metric_ok = _strip_quotes(params.get("metric", "")) == config.metric
            if decoder_ok and metric_ok:
                matches.append(entry)

        if not matches:
            raise FileNotFoundError(
                f"No decoder directory matched decoder={config.decoder}, "
                f"metric={config.metric} under {decoding_dir}"
            )
        if len(matches) > 1:
            paths = "\n  ".join(str(p) for p in matches)
            raise FileExistsError(
                f"Multiple decoder directories matched:\n  {paths}"
            )
        return matches[0]

    @staticmethod
    def _load_batch(
        bin_path: Path,
        le_path: Path,
    ) -> tuple[list[list[int]], list[bool], list[int]]:
        """Load one batch: binary flip indices + logical error flags, aligned by shot_id."""
        flip_by_shot = read_obs_flip_idx_file(bin_path)

        le_df = (
            pl.scan_parquet(le_path)
            .select(["shot_id", "is_logical_error"])
            .sort("shot_id")
            .collect()
        )

        n_binary = len(flip_by_shot)
        n_parquet = len(le_df)
        if n_binary != n_parquet:
            raise ValueError(
                f"Shot count mismatch: binary file has {n_binary} shots, "
                f"parquet has {n_parquet} shots ({le_path})"
            )

        shot_ids: list[int] = le_df["shot_id"].to_list()
        is_errors: list[bool] = le_df["is_logical_error"].to_list()

        return flip_by_shot, is_errors, shot_ids


# ---------------------------------------------------------------------------
# Histogram computation helpers
# ---------------------------------------------------------------------------


def _count_per_observable(
    flip_indices: list[list[int]],
    num_observables: int,
) -> np.ndarray:
    """Return per-observable appearance counts across all provided shots."""
    counts = np.zeros(num_observables, dtype=np.int64)
    for idxs in flip_indices:
        for i in idxs:
            if 0 <= i < num_observables:
                counts[i] += 1
    return counts


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ObsFlipAnalyzer:
    """Visualise observable flip index statistics.

    All plotting methods mutate ``ax`` in-place and return ``None``, matching
    the convention used throughout ``analysis/src/``.
    """

    def plot_histogram(
        self,
        data: ObsFlipData,
        ax: plt.Axes,
        *,
        separate_logical_error: bool = False,
        normalize: bool = False,
        bar_width: float = 0.35,
        color_all: str = "steelblue",
        color_error: str = "tomato",
        color_no_error: str = "mediumseagreen",
        alpha: float = 0.8,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Plot a bar chart of observable flip index frequencies.

        For each logical observable index (0 … num_observables−1) the bar shows
        how many shots had that index in their ``obs_flip_idx`` set.

        Parameters
        ----------
        data:
            Loaded :class:`ObsFlipData` returned by :class:`ObsFlipDataManager`.
        ax:
            Matplotlib axes to plot onto (mutated in-place).
        separate_logical_error:
            When ``True``, draw two side-by-side bars per observable: one for
            shots that are logical errors and one for shots that are not.
            When ``False``, draw a single bar for all shots combined.
        normalize:
            When ``True``, divide counts by the total number of shots in the
            respective group so the y-axis shows a fraction in [0, 1].
        bar_width:
            Width of each bar group.  Ignored when ``separate_logical_error=False``.
        color_all:
            Bar colour when ``separate_logical_error=False``.
        color_error:
            Bar colour for logical-error shots (``separate_logical_error=True``).
        color_no_error:
            Bar colour for non-error shots (``separate_logical_error=True``).
        alpha:
            Bar opacity.
        xlabel:
            X-axis label override (default: ``"Observable index"``).
        ylabel:
            Y-axis label override (default: ``"Count"`` or ``"Fraction of shots"``).
        title:
            Axes title.  ``None`` adds no title.
        """
        num_obs = data.num_observables
        if num_obs == 0:
            ax.text(0.5, 0.5, "No observables", ha="center", va="center",
                    transform=ax.transAxes)
            return

        x = np.arange(num_obs)

        if separate_logical_error:
            error_mask = data.is_logical_error
            no_error_mask = ~error_mask

            flip_error = [idxs for idxs, e in zip(data.flip_indices, error_mask) if e]
            flip_no_error = [idxs for idxs, e in zip(data.flip_indices, error_mask) if not e]

            counts_error = _count_per_observable(flip_error, num_obs)
            counts_no_error = _count_per_observable(flip_no_error, num_obs)

            n_error = max(error_mask.sum(), 1)
            n_no_error = max(no_error_mask.sum(), 1)

            if normalize:
                values_error = counts_error / n_error
                values_no_error = counts_no_error / n_no_error
            else:
                values_error = counts_error.astype(float)
                values_no_error = counts_no_error.astype(float)

            half = bar_width / 2
            ax.bar(x - half, values_no_error, width=bar_width, color=color_no_error,
                   alpha=alpha, label=f"No logical error (N={no_error_mask.sum()})")
            ax.bar(x + half, values_error, width=bar_width, color=color_error,
                   alpha=alpha, label=f"Logical error (N={error_mask.sum()})")
            ax.legend()
        else:
            counts = _count_per_observable(data.flip_indices, num_obs)
            n_shots = max(data.num_shots, 1)
            values = counts / n_shots if normalize else counts.astype(float)

            ax.bar(x, values, color=color_all, alpha=alpha,
                   label=f"All shots (N={data.num_shots})")
            ax.legend()

        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(num_obs)])
        ax.set_xlabel(xlabel if xlabel is not None else "Observable index")
        default_ylabel = "Fraction of shots" if normalize else "Count"
        ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)
        if title is not None:
            ax.set_title(title)
