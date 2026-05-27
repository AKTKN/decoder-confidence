#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import stim

_filter_import_error: Exception | None = None
try:
    from decoder_confidence.sampling.dem import filter_dem_by_basis
except Exception as exc:  # pragma: no cover - keep script runnable without package context
    filter_dem_by_basis = None
    _filter_import_error = exc


@dataclass(frozen=True)
class ErrorRecord:
    index: int
    line_no: Optional[int]
    line_text: Optional[str]
    prob: float
    comp_dets: List[List[int]]
    comp_obs: List[List[int]]
    xor_dets: List[int]


def _xor_sets(list_of_lists: Sequence[Sequence[int]]) -> List[int]:
    out: set[int] = set()
    for items in list_of_lists:
        current = set(items)
        out = (out - current) | (current - out)
    return sorted(out)


def _parse_error_targets(
    targets: Iterable[stim.DemTarget],
) -> Tuple[List[List[int]], List[List[int]]]:
    dets: List[List[int]] = [[]]
    obs: List[List[int]] = [[]]
    for t in targets:
        if t.is_separator():
            dets.append([])
            obs.append([])
            continue
        if t.is_relative_detector_id():
            dets[-1].append(int(t.val))
        elif t.is_logical_observable_id():
            obs[-1].append(int(t.val))
        else:
            raise ValueError("Unsupported DEM target type")
    return dets, obs


def _collect_error_lines(dem: stim.DetectorErrorModel) -> List[Tuple[int, str]]:
    lines = str(dem).splitlines()
    out: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if line.lstrip().startswith("error("):
            out.append((idx, line))
    return out


def _try_explain_errors(
    circuit: stim.Circuit, dem: stim.DetectorErrorModel
) -> Optional[List[str]]:
    if not hasattr(circuit, "explain_detector_error_model_errors"):
        return None
    try:
        explained = circuit.explain_detector_error_model_errors(dem)
    except TypeError:
        try:
            explained = circuit.explain_detector_error_model_errors()
        except Exception:
            return None
    except Exception:
        return None

    try:
        return [str(item) for item in explained]
    except Exception:
        return None


def _build_records(dem: stim.DetectorErrorModel) -> List[ErrorRecord]:
    error_lines = _collect_error_lines(dem)
    records: List[ErrorRecord] = []
    error_idx = 0

    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = float(inst.args_copy()[0])
        dets, obs = _parse_error_targets(inst.targets_copy())
        xor_dets = _xor_sets(dets)

        line_no = None
        line_text = None
        if error_idx < len(error_lines):
            line_no, line_text = error_lines[error_idx]
        error_idx += 1

        records.append(
            ErrorRecord(
                index=error_idx,
                line_no=line_no,
                line_text=line_text,
                prob=prob,
                comp_dets=dets,
                comp_obs=obs,
                xor_dets=xor_dets,
            )
        )

    return records


def _format_record(rec: ErrorRecord, explain: Optional[str]) -> str:
    comp_sizes = [len(c) for c in rec.comp_dets]
    obs_sizes = [len(c) for c in rec.comp_obs]
    lines: List[str] = []
    lines.append("-" * 80)
    lines.append(f"error_index={rec.index} line={rec.line_no} p={rec.prob}")
    lines.append(f"component_det_sizes={comp_sizes} component_obs_sizes={obs_sizes}")
    lines.append(f"xor_det_count={len(rec.xor_dets)} xor_dets={rec.xor_dets}")
    if rec.line_text:
        lines.append(f"dem_line: {rec.line_text}")
    lines.append(f"component_dets: {rec.comp_dets}")
    lines.append(f"component_obs: {rec.comp_obs}")
    if explain:
        lines.append("explain:")
        lines.append(explain)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect DEM errors with >=3 detectors and print details."
    )
    parser.add_argument(
        "--circuit",
        required=True,
        help="Path to the .stim circuit file",
    )
    parser.add_argument(
        "--decompose_errors",
        default=False,
        action="store_true",
        help="Use decompose_errors=True when building DEM",
    )
    parser.add_argument(
        "--min_detectors",
        type=int,
        default=3,
        help="Minimum detector count to report (default: 3)",
    )
    parser.add_argument(
        "--remove_basis",
        default=None,
        choices=("X", "Z"),
        help="If set, filter DEM to remove detectors in this basis (X or Z)",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=50,
        help="Max number of records to print to stdout (default: 50)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file for full details",
    )
    parser.add_argument(
        "--explain",
        default=False,
        action="store_true",
        help="Try to print circuit-level explanations if available",
    )

    args = parser.parse_args()
    circuit_path = Path(args.circuit)
    if not circuit_path.exists():
        raise FileNotFoundError(f"circuit not found: {circuit_path}")

    circuit = stim.Circuit.from_file(str(circuit_path))
    dem = circuit.detector_error_model(decompose_errors=bool(args.decompose_errors))

    if args.remove_basis:
        if filter_dem_by_basis is None:
            detail = f" ({_filter_import_error})" if _filter_import_error else ""
            raise RuntimeError(
                "filter_dem_by_basis import failed; run from repo root" + detail
            )
        dem = filter_dem_by_basis(dem, args.remove_basis)

    records = _build_records(dem)
    explain_list = _try_explain_errors(circuit, dem) if args.explain else None

    total = len(records)
    min_det = int(args.min_detectors)
    if min_det < 1:
        raise ValueError("min_detectors must be >= 1")

    flagged: List[ErrorRecord] = []
    for rec in records:
        if len(rec.xor_dets) >= min_det:
            flagged.append(rec)

    comp_flagged = [
        rec
        for rec in records
        if any(len(comp) >= min_det for comp in rec.comp_dets)
    ]

    header_lines = [
        "=== DEM hyperedge inspection ===",
        f"circuit: {circuit_path}",
        f"decompose_errors: {bool(args.decompose_errors)}",
        f"total_errors: {total}",
        f"xor_det_count >= {min_det}: {len(flagged)}",
        f"any_component_det_count >= {min_det}: {len(comp_flagged)}",
    ]
    print("\n".join(header_lines))

    if not flagged:
        print("No errors with xor_det_count >= min_detectors.")
        return 0

    max_records = int(args.max_records)
    if max_records < 0:
        raise ValueError("max_records must be >= 0")

    output_path = Path(args.output) if args.output else None
    output_handle = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")
        output_handle.write("\n".join(header_lines) + "\n")

    for idx, rec in enumerate(flagged, start=1):
        explain = None
        if explain_list and rec.index - 1 < len(explain_list):
            explain = explain_list[rec.index - 1]
        formatted = _format_record(rec, explain)

        if output_handle is not None:
            output_handle.write(formatted + "\n")

        if max_records == 0:
            continue
        if idx <= max_records:
            print(formatted)

    if output_handle is not None:
        output_handle.close()
        print(f"full_output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
