from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from analysis.src.config import PlotConfig, normalize_metric_names

# Regex to extract the batch index from filenames like "metric=logical_gap_batch=5.parquet"
# or "logicalerror_batch=5.parquet".
_BATCH_RE = re.compile(r"_batch=(\d+)\.parquet$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_kv(name: str) -> dict[str, str]:
    """Split a comma-separated ``key=value`` directory name into a dict.

    Values produced by ``repr()`` (e.g. ``"'Ratio'"`` with surrounding single
    quotes) are returned as-is; callers strip quotes when needed.
    """
    result: dict[str, str] = {}
    for part in name.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        result[key.strip()] = val.strip()
    return result


def _strip_quotes(s: str) -> str:
    """Remove surrounding single or double quotes produced by ``repr()``."""
    return s.strip("'\"")


def _matches_filter(raw_val: str, expected: Any) -> bool:
    """Return True when *raw_val* (a string from a directory name) equals *expected*.

    Type coercion is performed so callers can pass typed Python values in
    :attr:`PlotConfig.filters`.
    """
    clean = _strip_quotes(raw_val)
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


def _matches_all_filters(params: dict[str, str], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if key not in params:
            # Some historical/default-false directory flags are omitted from
            # path names.  Treat a missing boolean flag as False so filters like
            # {"ibm_reproduce": False} match the default directory layout,
            # while {"ibm_reproduce": True} still requires an explicit flag.
            if isinstance(expected, bool) and expected is False:
                continue
            return False
        if not _matches_filter(params[key], expected):
            return False
    return True


def _infer_literal(raw: str) -> pl.Expr:
    """Build a :func:`polars.lit` expression with an inferred type from *raw*.

    Precedence: bool → int (Int64) → float (Float64) → str (Utf8).
    Surrounding quotes from ``repr()`` are stripped first.
    """
    s = _strip_quotes(raw)
    if s.lower() in ("true", "false"):
        return pl.lit(s.lower() == "true")
    try:
        return pl.lit(int(s), dtype=pl.Int64)
    except ValueError:
        pass
    try:
        return pl.lit(float(s), dtype=pl.Float64)
    except ValueError:
        pass
    return pl.lit(s)


def _extract_batch_idx(filename: str) -> int | None:
    m = _BATCH_RE.search(filename)
    return int(m.group(1)) if m else None


def _is_circuit_params_dir(name: str) -> bool:
    """Heuristic: a circuit-params directory must contain at least code, d, p."""
    kv = _parse_kv(name)
    return "code" in kv and "d" in kv and "p" in kv


def _use_linearized_logical_error(config: PlotConfig, metric_name: str) -> bool:
    flag = bool(
        config.plot_params.get("use_linearize", False)
        or config.extra_options.get("use_linearize", False)
    )
    if flag and metric_name != "forced_gap_ml":
        raise ValueError(
            "use_linearize is only supported when metric_name='forced_gap_ml'"
        )
    return flag


# ---------------------------------------------------------------------------
# SimulationDataManager
# ---------------------------------------------------------------------------

class SimulationDataManager:
    """Queries simulation result directories and returns a Polars LazyFrame.

    Directory layout expected::

        result_dir_root/
        └── <circuit_params>/          # e.g. code=surface_code_Z,d=5,p=0.001,...
            └── decoding_result/
                └── <decoder_params>/  # e.g. decoder=ILP,metric=logical_gap
                    ├── metric=<name>_batch=<N>.parquet
                    └── logicalerror_batch=<N>.parquet

    Parameters
    ----------
    result_dir_root:
        Root directory that contains circuit-parameter subdirectories.
    """

    def __init__(self, result_dir_root: Path) -> None:
        self.result_dir_root = Path(result_dir_root)
        if not self.result_dir_root.is_dir():
            raise FileNotFoundError(f"result_dir_root not found: {self.result_dir_root}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, config: PlotConfig) -> pl.LazyFrame:
        """Scan directories matching *config* and return a concatenated LazyFrame.

        The returned LazyFrame is built lazily using :func:`polars.scan_parquet`.
        The metric values and logical-error flags (stored in separate parquet
        files) are joined on ``shot_id``.  Path-encoded parameters (``p``, ``d``,
        ``decoder``, etc.) are appended as literal columns.

        No data is loaded into memory until ``.collect()`` is called downstream.

        Raises
        ------
        FileNotFoundError
            If no matching data is found for the given config.
        """
        frames: list[pl.LazyFrame] = []
        metric_names = normalize_metric_names(config.metric_name)

        for metric_name in metric_names:
            for circuit_dir, circuit_params in self._iter_circuit_dirs(config.filters):
                for dm_dir, dm_params in self._iter_decoder_dirs(
                    circuit_dir, config, metric_name
                ):
                    all_params = {**circuit_params, **dm_params}
                    frames.extend(
                        self._build_frames_for_dir(
                            dm_dir, config, all_params, metric_name
                        )
                    )

        if not frames:
            metrics = ", ".join(metric_names)
            raise FileNotFoundError(
                f"No parquet data found for metric(s)='{metrics}' "
                f"with filters={config.filters} under {self.result_dir_root}"
            )

        return pl.concat(frames, how="diagonal")

    # ------------------------------------------------------------------
    # Internal iteration helpers
    # ------------------------------------------------------------------

    def _iter_circuit_dirs(
        self, filters: dict[str, Any]
    ) -> Iterator[tuple[Path, dict[str, str]]]:
        """Yield ``(path, parsed_params)`` for each matching circuit-params directory."""
        for entry in sorted(self.result_dir_root.iterdir()):
            if not entry.is_dir():
                continue
            if not _is_circuit_params_dir(entry.name):
                continue
            params = _parse_kv(entry.name)
            if not _matches_all_filters(params, filters):
                continue
            yield entry, params

    def _iter_decoder_dirs(
        self, circuit_dir: Path, config: PlotConfig, metric_name: str
    ) -> Iterator[tuple[Path, dict[str, str]]]:
        """Yield ``(path, parsed_params)`` for each matching decoder/metric directory."""
        decoding_dir = circuit_dir / "decoding_result"
        if not decoding_dir.is_dir():
            return

        for entry in sorted(decoding_dir.iterdir()):
            if not entry.is_dir():
                continue
            params = _parse_kv(entry.name)

            # Must match the requested metric name
            if _strip_quotes(params.get("metric", "")) != metric_name:
                continue

            # Must match one of the requested decoders (if specified)
            decoder = _strip_quotes(params.get("decoder", ""))
            if config.decoder_names is not None and decoder not in config.decoder_names:
                continue

            yield entry, params

    def _build_frames_for_dir(
        self,
        dm_dir: Path,
        config: PlotConfig,
        all_params: dict[str, str],
        metric_name: str,
    ) -> list[pl.LazyFrame]:
        """Return one LazyFrame per batch file found in *dm_dir*."""
        frames: list[pl.LazyFrame] = []

        for metric_file in sorted(dm_dir.glob(f"metric={metric_name}_batch=*.parquet")):
            batch_idx = _extract_batch_idx(metric_file.name)
            if batch_idx is None:
                continue

            # Honour the optional batch-index filter
            if config.batch_indices is not None and batch_idx not in config.batch_indices:
                continue

            le_file = self._logicalerror_file_for_metric(
                dm_dir=dm_dir,
                config=config,
                metric_name=metric_name,
                batch_idx=batch_idx,
            )
            if not le_file.exists():
                continue

            lf = self._build_single_frame(
                metric_file,
                le_file,
                all_params,
                batch_idx,
                metric_name,
            )
            frames.append(lf)

        return frames

    def _build_single_frame(
        self,
        metric_file: Path,
        le_file: Path,
        params: dict[str, str],
        batch_idx: int,
        metric_name: str,
    ) -> pl.LazyFrame:
        """Lazily scan one batch of metric + logical-error parquet files and join them.

        Parameters inferred from directory names are appended as literal columns so
        downstream ``group_by`` / ``partition_by`` operations work without loading the
        full dataset.
        """
        metric_lf = pl.scan_parquet(metric_file).select(["shot_id", metric_name])
        le_lf = pl.scan_parquet(le_file).select(["shot_id", "is_logical_error"])

        # Join on shot_id — both files share this key
        combined = metric_lf.join(le_lf, on="shot_id", how="inner")

        # Append path-encoded metadata as literal columns
        literal_exprs = [
            _infer_literal(val).alias(key) for key, val in params.items()
        ]
        literal_exprs.append(pl.lit(batch_idx, dtype=pl.Int64).alias("batch"))

        return combined.with_columns(literal_exprs)

    def _logicalerror_file_for_metric(
        self,
        dm_dir: Path,
        config: PlotConfig,
        metric_name: str,
        batch_idx: int,
    ) -> Path:
        if not _use_linearized_logical_error(config, metric_name):
            return dm_dir / f"logicalerror_batch={batch_idx}.parquet"

        linearize_dir = self._find_linearize_metric_dir(dm_dir)
        linearize_le_file = linearize_dir / f"logicalerror_batch={batch_idx}.parquet"
        if not linearize_le_file.exists():
            raise FileNotFoundError(
                "use_linearize=True requested, but no matching "
                f"linearize_logicalgap logicalerror file was found for batch={batch_idx} "
                f"under {linearize_dir}"
            )
        return linearize_le_file

    def _find_linearize_metric_dir(self, dm_dir: Path) -> Path:
        decoding_dir = dm_dir.parent
        current_params = _parse_kv(dm_dir.name)
        decoder = _strip_quotes(current_params.get("decoder", ""))

        candidates: list[Path] = []
        for entry in sorted(decoding_dir.iterdir()):
            if not entry.is_dir():
                continue
            params = _parse_kv(entry.name)
            if _strip_quotes(params.get("metric", "")) != "linearize_logicalgap":
                continue
            if _strip_quotes(params.get("decoder", "")) != decoder:
                continue
            candidates.append(entry)

        if not candidates:
            raise FileNotFoundError(
                "use_linearize=True requested, but no sibling "
                f"linearize_logicalgap directory was found for decoder={decoder!r} "
                f"next to {dm_dir}"
            )
        if len(candidates) > 1:
            raise FileNotFoundError(
                "use_linearize=True requested, but multiple sibling "
                f"linearize_logicalgap directories were found for decoder={decoder!r}: "
                + ", ".join(str(path) for path in candidates)
            )
        return candidates[0]
