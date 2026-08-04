"""Conditional statistics of the anchored forced gap Delta_anc, for slides.

Uses existing circuit-level simulation data only (no re-simulation). Reuses
:func:`analysis.src.anchored_reselection.load_anchored_reselection_lazy` to
load the ``forced_gap_ml`` detailed-stat table (``get_detail_stat=True``),
which already carries, per shot:

* ``forced_stage1_weight``    -- w(e^(0)), the baseline (stage-1) weight
* ``forced_stage1_obs_flip``  -- y0 = 1[baseline is a logical error]
* ``forced_stage2_weight``    -- min_{i=1..k} w(e^(i)), the lightest
  stage-2 (forced) candidate's weight
* ``forced_stage2_obs_flip``  -- y2 = 1[that lightest stage-2 candidate is
  a logical error]

Delta_anc = forced_stage2_weight - forced_stage1_weight, matching the
``linearize_logicalgap`` metric / the anchored-forced-gap curve in the
existing post-selection figure (post_selection_ler_cln_p0003.png).

Run from the repository root::

    python -m analysis.circuit_level.anchored_gap_conditional_stats
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from analysis.src.anchored_reselection import (
    AnchoredReselectionConfig,
    load_anchored_reselection_lazy,
)
from analysis.src.confidence import wilson_ci
from analysis.src.data_manager import (
    SimulationDataManager,
    _is_circuit_params_dir,
    _matches_all_filters,
    _parse_kv,
    _strip_quotes,
)
from analysis.src.stage_gap_difference import _find_detail_dirs

RESULT_DIR = Path(__file__).parents[2] / "simulation_data"
OUT_DIR = Path(__file__).parent / "figs"
OUT_JSON = OUT_DIR / "anchored_gap_conditional_stats.json"

DECODER_NAMES = ["BP-LSD"]

# Same circuit settings as the anchored-forced-gap curve in
# post_selection_ler_cln_p0003.png (BASE_FILTERS in circuit_level.ipynb).
TARGET_FILTERS: dict[str, Any] = {
    "d": 6,
    "p": 0.003,
    "noisemodel": "uniform",
    "code": "bivariate_bicycle_code_Z",
    "rounds": 6,
}
SWEEP_FILTERS_BASE: dict[str, Any] = {
    k: v for k, v in TARGET_FILTERS.items() if k != "p"
}

REQUIRED_COLUMNS = (
    "forced_stage1_weight",
    "forced_stage1_obs_flip",
    "forced_stage2_weight",
    "forced_stage2_obs_flip",
)

ALPHA = 0.05  # 95% Wilson CI


def discover_available_p_values(
    result_dir_root: Path,
    filters_base: dict[str, Any],
    decoder_names: list[str],
) -> list[float]:
    """Return every ``p`` for which forced_gap_ml,get_detail_stat=True data exists.

    Scans circuit-parameter directories matching *filters_base* (every key
    of :data:`TARGET_FILTERS` except ``p``) and keeps a ``p`` value only if
    at least one matching decoder directory has ``get_detail_stat=True``.
    """
    p_values: set[float] = set()
    for circuit_dir in sorted(result_dir_root.iterdir()):
        if not circuit_dir.is_dir() or not _is_circuit_params_dir(circuit_dir.name):
            continue
        params = _parse_kv(circuit_dir.name)
        if not _matches_all_filters(params, filters_base):
            continue
        if "p" not in params:
            continue
        detail_dirs = _find_detail_dirs(circuit_dir, "forced_gap_ml", decoder_names)
        if detail_dirs:
            p_values.add(float(_strip_quotes(params["p"])))
    return sorted(p_values)


@dataclass
class ConditionalStats:
    p: float
    n_loaded: int
    n_excluded: int
    n_total: int
    n_neg: int
    n_zero: int
    n_pos: int
    q: float
    q_ci: tuple[float, float]
    eps: float
    eps_ci: tuple[float, float]
    eps_minus: float
    eps_minus_ci: tuple[float, float]
    eps_plus: float
    eps_plus_ci: tuple[float, float]
    concentration: float
    concentration_ci: tuple[float, float]
    p_y2_given_neg: float
    p_y2_given_neg_ci: tuple[float, float]
    consistency_lhs: float
    consistency_rhs: float
    consistency_ok: bool

    def to_dict(self) -> dict[str, Any]:
        d = {}
        for key, val in self.__dict__.items():
            if isinstance(val, tuple):
                d[f"{key}_low"], d[f"{key}_high"] = val
            else:
                d[key] = val
        return d


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo, hi = wilson_ci(k, n, alpha=ALPHA)
    return float(lo), float(hi)


def _ratio(k: int, n: int) -> float:
    return float(k) / n if n > 0 else float("nan")


def load_shot_table(
    result_dir_root: Path, filters: dict[str, Any], decoder_names: list[str]
) -> pl.DataFrame:
    config = AnchoredReselectionConfig(decoder_names=decoder_names, filters=filters)
    lf = load_anchored_reselection_lazy(result_dir_root, config)
    return lf.collect()


def compute_conditional_stats(df: pl.DataFrame, p: float) -> ConditionalStats:
    n_loaded = df.height

    missing_mask = np.zeros(n_loaded, dtype=bool)
    for col in REQUIRED_COLUMNS:
        arr = df[col].to_numpy()
        if arr.dtype.kind == "f":
            missing_mask |= np.isnan(arr)
        missing_mask |= df[col].is_null().to_numpy()
    n_excluded = int(missing_mask.sum())

    clean = df.filter(pl.Series(~missing_mask))
    n = clean.height

    w0 = clean["forced_stage1_weight"].to_numpy().astype(float)
    w2min = clean["forced_stage2_weight"].to_numpy().astype(float)
    y0 = clean["forced_stage1_obs_flip"].to_numpy().astype(bool)
    y2 = clean["forced_stage2_obs_flip"].to_numpy().astype(bool)

    delta_anc = w2min - w0
    neg_mask = delta_anc < 0
    zero_mask = delta_anc == 0
    pos_mask = delta_anc >= 0  # includes the boundary Delta_anc == 0

    n_neg = int(neg_mask.sum())
    n_zero = int(zero_mask.sum())
    n_pos = int(pos_mask.sum())

    n_y0 = int(y0.sum())
    n_y0_and_neg = int((y0 & neg_mask).sum())
    n_y0_given_neg = int(y0[neg_mask].sum()) if n_neg > 0 else 0
    n_y0_given_pos = int(y0[pos_mask].sum()) if n_pos > 0 else 0
    n_y2_given_neg = int(y2[neg_mask].sum()) if n_neg > 0 else 0

    q = _ratio(n_neg, n)
    eps = _ratio(n_y0, n)
    eps_minus = _ratio(n_y0_given_neg, n_neg)
    eps_plus = _ratio(n_y0_given_pos, n_pos)
    p_y2_given_neg = _ratio(n_y2_given_neg, n_neg)

    # concentration = q * eps_minus / eps = P(Delta_anc<0, y0=1) / P(y0=1)
    #               = P(Delta_anc<0 | y0=1); compute directly as a proportion
    # so its Wilson CI is a genuine binomial CI rather than a delta-method
    # approximation propagated through a ratio of estimates.
    concentration = _ratio(n_y0_and_neg, n_y0)

    consistency_lhs = q * eps_minus + (1.0 - q) * eps_plus
    consistency_rhs = eps
    consistency_ok = bool(np.isclose(consistency_lhs, consistency_rhs, rtol=1e-9, atol=1e-9))

    return ConditionalStats(
        p=p,
        n_loaded=n_loaded,
        n_excluded=n_excluded,
        n_total=n,
        n_neg=n_neg,
        n_zero=n_zero,
        n_pos=n_pos,
        q=q,
        q_ci=_wilson(n_neg, n),
        eps=eps,
        eps_ci=_wilson(n_y0, n),
        eps_minus=eps_minus,
        eps_minus_ci=_wilson(n_y0_given_neg, n_neg),
        eps_plus=eps_plus,
        eps_plus_ci=_wilson(n_y0_given_pos, n_pos),
        concentration=concentration,
        concentration_ci=_wilson(n_y0_and_neg, n_y0),
        p_y2_given_neg=p_y2_given_neg,
        p_y2_given_neg_ci=_wilson(n_y2_given_neg, n_neg),
        consistency_lhs=consistency_lhs,
        consistency_rhs=consistency_rhs,
        consistency_ok=consistency_ok,
    )


def _fmt_pct_ci(value: float, ci: tuple[float, float]) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value:.4%} [{ci[0]:.4%}, {ci[1]:.4%}]"


def print_stats_table(stats: ConditionalStats) -> None:
    print(f"\n=== Anchored forced gap conditional statistics (p={stats.p:g}) ===")
    rows = [
        ("N (loaded)", str(stats.n_loaded)),
        ("N (excluded: missing/non-converged)", str(stats.n_excluded)),
        ("N (used)", str(stats.n_total)),
        ("N(Delta_anc < 0)", f"{stats.n_neg}"),
        ("N(Delta_anc = 0)  [boundary, reported separately]", f"{stats.n_zero}"),
        ("N(Delta_anc >= 0)", f"{stats.n_pos}"),
        ("q = P(Delta_anc < 0)", _fmt_pct_ci(stats.q, stats.q_ci)),
        ("eps = P(y0=1)", _fmt_pct_ci(stats.eps, stats.eps_ci)),
        ("eps_minus = P(y0=1 | Delta_anc < 0)", _fmt_pct_ci(stats.eps_minus, stats.eps_minus_ci)),
        ("eps_plus = P(y0=1 | Delta_anc >= 0)", _fmt_pct_ci(stats.eps_plus, stats.eps_plus_ci)),
        (
            "concentration = q*eps_minus/eps = P(Delta_anc<0 | y0=1)",
            _fmt_pct_ci(stats.concentration, stats.concentration_ci),
        ),
        ("P(y2=1 | Delta_anc < 0)", _fmt_pct_ci(stats.p_y2_given_neg, stats.p_y2_given_neg_ci)),
    ]
    name_w = max(len(r[0]) for r in rows)
    for name, val in rows:
        print(f"  {name.ljust(name_w)} : {val}")

    print(
        f"\n  consistency check: q*eps_minus + (1-q)*eps_plus = {stats.consistency_lhs:.6g} "
        f"vs eps = {stats.consistency_rhs:.6g} -> "
        f"{'PASS' if stats.consistency_ok else 'FAIL'}"
    )


def slide_sentence(stats: ConditionalStats) -> str:
    return (
        f"Δ_anc < 0 のショットは全体の {stats.q:.1%} にすぎないが、"
        f"論理エラーの {stats.concentration:.0%} がこの領域に集中する"
    )


def main() -> None:
    manager = SimulationDataManager(RESULT_DIR)

    print(f"Result root: {RESULT_DIR}")
    print(f"Target filters (matches post_selection_ler_cln_p0003.png BASE_FILTERS): {TARGET_FILTERS}")
    print(f"Decoder: {DECODER_NAMES}")

    # --- Primary point: exactly the figure's configuration ---
    df_target = load_shot_table(manager.result_dir_root, TARGET_FILTERS, DECODER_NAMES)
    target_stats = compute_conditional_stats(df_target, p=float(TARGET_FILTERS["p"]))
    print_stats_table(target_stats)
    print("\nSlide sentence (JA):")
    print(f"  {slide_sentence(target_stats)}")

    # --- Secondary: sweep over every p with matching detail-stat data ---
    available_p = discover_available_p_values(
        manager.result_dir_root, SWEEP_FILTERS_BASE, DECODER_NAMES
    )
    print(f"\nAvailable p values with get_detail_stat=True forced_gap_ml data "
          f"under {SWEEP_FILTERS_BASE}: {available_p}")

    sweep_stats: list[ConditionalStats] = []
    for p in available_p:
        filters = {**SWEEP_FILTERS_BASE, "p": p}
        df_p = load_shot_table(manager.result_dir_root, filters, DECODER_NAMES)
        s = compute_conditional_stats(df_p, p=p)
        sweep_stats.append(s)
        if p != TARGET_FILTERS["p"]:
            print_stats_table(s)

    if len(available_p) <= 1:
        print(
            "\nNote: only one p value has get_detail_stat=True forced_gap_ml data "
            "under this circuit configuration (no physical-error-rate sweep exists "
            "for the circuit-level/uniform-noise setting), so the 'all settings' "
            "table below has a single row."
        )

    print("\n=== Summary table across available p ===")
    header = (
        f"{'p':>8} | {'N':>9} | {'excluded':>8} | {'q':>18} | {'eps':>18} | "
        f"{'eps_minus':>18} | {'eps_plus':>18} | {'concentration':>18} | {'P(y2=1|neg)':>18} | consistency"
    )
    print(header)
    print("-" * len(header))
    for s in sweep_stats:
        print(
            f"{s.p:>8.4g} | {s.n_total:>9d} | {s.n_excluded:>8d} | "
            f"{s.q:>17.4%} | {s.eps:>17.4%} | {s.eps_minus:>17.4%} | "
            f"{s.eps_plus:>17.4%} | {s.concentration:>17.4%} | {s.p_y2_given_neg:>17.4%} | "
            f"{'PASS' if s.consistency_ok else 'FAIL'}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_dir_root": str(RESULT_DIR),
        "decoder_names": DECODER_NAMES,
        "target_filters": TARGET_FILTERS,
        "sweep_filters_base": SWEEP_FILTERS_BASE,
        "alpha": ALPHA,
        "target": target_stats.to_dict(),
        "sweep": [s.to_dict() for s in sweep_stats],
        "slide_sentence_ja": slide_sentence(target_stats),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
