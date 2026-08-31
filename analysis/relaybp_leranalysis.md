# Why Reselection Underperforms Anchoring Under RELAY-BP: A Post-Selection Case-Breakdown Analysis

## Abstract

For the `forced_gap_ml` confidence metric, the decoder re-selects its final answer to be
the minimum-weight solution found across a baseline decode and `K` constrained
("forced") re-decodes, one per logical observable. A closely related metric,
`linearize_logicalgap`, uses the exact same two-stage decoding process but *always*
keeps the baseline (stage-1) answer as the final prediction, using the constrained
re-decodes only to score confidence. We refer to the former as the **ordinary forced
gap** (reselecting) and the latter as the **anchored forced gap** (non-reselecting).

Under the BP-LSD decoder, reselection is a net win: at the post-selection operating
point that discards nothing beyond outright decode failures ("0-abort" point), the
ordinary forced gap achieves a *lower* logical error rate (LER) than the anchored
variant (1.643% vs. 1.781%). Under RELAY-BP on the same code, noise model, and
physical error rate, the effect **reverses**: the ordinary forced gap is *worse* than
the anchored variant at the same operating point (1.256% vs. 0.719%, a 1.75x
degradation). This report (1) confirms that this reversal is not a simulation or
post-selection-pipeline defect, (2) reconstructs the per-shot decision-outcome
breakdown that produces it, and (3) identifies the mechanistic cause in RELAY-BP's
`stop_nconv=1` relay-leg acceptance policy interacting with a highly degenerate
`[[72,12,6]]` bivariate-bicycle code.

## 1. Background

Both metrics share the same two-stage decoding structure
(`src/decoder_confidence/decoding/_forced_gap.py`,
`src/decoder_confidence/decoding/_linearize_logicalgap.py`):

- **Stage 1 (baseline).** Decode the syndrome unconstrained, obtaining correction
  `e^(0)`, weight `w_0`, and logical class `lambda_0`.
- **Stage 2 (forced).** For each of the `K = num_observables` logical bits `i`,
  append observable row `i` to the parity-check matrix with target value
  `1 - lambda_0[i]`, forcing the decode into the complementary class for bit `i`.
  This yields candidates `e^(1), ..., e^(K)` with weights `w_1, ..., w_K` (only
  instances that converge are kept, relevant for RELAY-BP).

The two metrics differ only in what they do with these `K+1` candidates:

| Metric | Final prediction | Confidence value |
|---|---|---|
| `forced_gap_ml` (ordinary / reselecting) | `argmin` weight over **all** `K+1` candidates | gap to nearest *distinct-class* competitor |
| `linearize_logicalgap` (anchored / non-reselecting) | always `e^(0)` (stage 1) | `min_i(w_i) - w_0` |

Because the anchored prediction is fixed to stage 1, its LER equals
`P(lambda_0 != lambda_true)` unconditionally. The ordinary metric's LER differs from
it only on shots where reselection actually changes the winner, i.e. where some
`w_i < w_0`.

## 2. Experimental Setup

All data is drawn from `simulation_result/` under:

```
code=bivariate_bicycle_code_Z,d=6,rounds=6,noisemodel=uniform,p=0.003,
use_both=False,ibm_reproduce=True,xyz=False
```

(matching `BASE_FILTERS` in `analysis/circuit_level/circuit_level.ipynb`).

**Problem size** (from run metadata, identical for both decoders):
check-matrix columns = 2232, check-matrix rows (detectors) = 252,
logical observables `K` = 12. Each shot set has 10 batches x 100,000 shots =
**1,000,000 shots** per decoder/metric directory.

**Decoder configurations** (`decoder_config.decoder_options` in each run's
`metadata/*.json`):

| Parameter | BP-LSD | RELAY-BP |
|---|---|---|
| `bp_method` | `min_sum` | — |
| `lsd_method` / `lsd_order` | `LSD_CS` / 0 | — |
| `max_iter` | 30 | — |
| `ms_scaling_factor` | 1.0 | — |
| `gamma0` | — | 0.1 |
| `gamma_dist_interval` | — | [-0.19, 0.26] |
| `num_sets` (max relay legs) | — | 300 |
| `pre_iter` | — | 80 |
| `set_max_iter` | — | 60 |
| `stop_nconv` | — | **1** |
| `seed` | — | 0 |
| Max possible iterations | 30 | `pre_iter + num_sets*set_max_iter` = **18,080** |

Both `forced_gap_ml` and `linearize_logicalgap` directories for RELAY-BP were run
with `forced_unconverged_confidence_value='positive'` and `get_detail_stat=True`
against the *same* sampled detector-event file and the same `seed=0`, i.e. the two
runs decode identical shots.

**Wall-clock cost.** BP-LSD averages 0.51 s/shot/worker (~9 min per 100k-shot
batch on 96 workers); RELAY-BP averages 27.3 s/shot/worker (~8.0 h per batch),
roughly 54x slower, entirely attributable to Stage-2 (see Section 5).

## 3. Methods

### 3.1 Simulation-logic verification

Both metric implementations were read line-by-line
(`src/decoder_confidence/decoding/_forced_gap.py`,
`src/decoder_confidence/decoding/_linearize_logicalgap.py`) and cross-checked
against the RELAY-BP non-convergence design note
(`documents/relay_bp_nonconvergence_behavior.md`) and its accompanying unit tests
(`tests/test_relay_bp_nonconvergence.py`). No discrepancy between intended and
implemented behavior was found. In particular:

- Stage-2 candidates are correctly restricted to converged instances only
  (`(not self._relay_bp_adapter) or converged2` gating, `_forced_gap.py`).
- Stage-1 non-convergence forces `is_logical_error=True` and skips Stage 2
  entirely, identically in both metrics.
- A direct comparison of the `forced_gap_ml` and `linearize_logicalgap` RELAY-BP
  result directories (batch 1) confirms **bit-exact agreement** of the Stage-1
  baseline decode (`baseline_correction_weight`, `baseline_logical_error`, and the
  non-convergent-shot set are identical across the two independently-executed
  runs), ruling out non-determinism as a contributing factor.

### 3.2 Case reconstruction from `detailed_stats`

The BP-LSD `forced_gap_ml` directory was run with `get_all_failure_rate=True`,
which records an explicit 5-way `forced_gap_ml_case` label per shot
(`_forced_gap.py`, lines 33-49). The RELAY-BP run predates that flag being enabled,
so its per-shot decision outcome was reconstructed from the `detailed_stats` /
`decoder_stat` / `is_logical_error` columns that *are* present:

```
Y0    = baseline_logical_error        (stage-1 prediction wrong?)
Ystar = is_logical_error              (final, reselected prediction wrong?)

case -1 ("kept")    : Y0=0, Ystar=0   (stage-1 correct, kept)
case  1 ("fixed")   : Y0=1, Ystar=0   (stage-1 wrong, reselection rescued it)
case  3 ("broken")  : Y0=0, Ystar=1   (stage-1 correct, reselection broke it)
case 0+2 ("still wrong") : Y0=1, Ystar=1   (stage-1 wrong, still wrong after reselection)
```

This reconstruction cannot separate case 0 (no correct alternative existed among the
12 Stage-2 branches) from case 2 (a correct alternative existed but a different,
lower-weight *wrong* branch was adopted instead), since only the best and
second-best Stage-2 candidates are retained on disk. It **can** exactly recover
cases -1, 1, and 3, which are the only three cases whose outcome differs between the
anchored and ordinary metrics. This reconstruction was validated by applying the
identical procedure to BP-LSD (which has the ground-truth label) and obtaining an
**exact match** against the true `forced_gap_ml_case` counts (Table 1), confirming
the method before applying it to RELAY-BP.

Shots where the RELAY-BP Stage-1 (baseline) decode itself failed to converge
(`baseline_iteration` is `NaN`) are excluded from the case breakdown and treated
separately, matching the "0-abort" reference point defined by
`PostSelectSpec(separate_unconverged=True)` in `analysis/src/postselect.py`
(dashed reference line in the notebook's post-selection plots): it discards exactly
the non-convergent shots and none else.

## 4. Results

### 4.1 Case breakdown

**Table 1 — Per-shot decision-outcome breakdown** (N = 1,000,000 shots each;
RELAY-BP percentages are relative to the 999,425 shots whose Stage-1 decode
converged).

| Case | Meaning | BP-LSD (actual `forced_gap_ml_case`) | RELAY-BP (reconstructed) |
|---|---|---:|---:|
| -1 | Stage-1 correct, kept | 971,907 (97.191%) | 983,707 (98.427%) |
| 1 | Stage-1 wrong -> **rescued** by reselection | 11,664 (**1.166%**) | 3,164 (**0.317%**) |
| 3 | Stage-1 correct -> **broken** by reselection | 10,285 (**1.029%**) | 8,531 (**0.854%**) |
| 0 + 2 | Stage-1 wrong, still wrong after reselection | 1,997 + 4,147 = 6,144 (0.614%) | 4,023 (0.402%) |
| Stage-1 non-convergent (excluded above) | — | n/a | 575 (0.058%) |

Cases -1 and (0+2) contribute identically to both metrics' LER and cancel out of
the comparison; the sign of `LER_forced - LER_anchored` is set entirely by
`count(case 3) - count(case 1)`:

- BP-LSD: 10,285 - 11,664 = **-1,379** -> forced gap *improves* LER.
- RELAY-BP: 8,531 - 3,164 = **+5,367** -> forced gap *degrades* LER, and by a
  much larger margin (relative to population size, ~2.7x more "broken" than
  "fixed" events, versus BP-LSD's near-parity ~0.88x).

### 4.2 LER at the 0-abort operating point

**Table 2 — Post-selected LER at the "reject only outright decode failures"
point** (dashed reference line in `circuit_level.ipynb`, cells 24-25).

| Decoder | Anchored (`linearize_logicalgap`) LER | Ordinary (`forced_gap_ml`) LER | Ratio (forced / anchored) |
|---|---:|---:|---:|
| BP-LSD | 1.781% | 1.643% | 0.923 (7.7% better) |
| RELAY-BP | 0.719% | 1.256% | **1.747 (75% worse)** |

These numbers reproduce, from first principles via the raw per-shot tables, the
qualitative pattern visible in the notebook's post-selection curves and quantify
its magnitude.

### 4.3 Mechanistic diagnostics (RELAY-BP only, restricted to shots where a
Stage-2 candidate outweighed Stage-1, i.e. `Delta_anc = min_i(w_i) - w_0 < 0`,
`n_A` = 12,182 converged shots)

**Table 3 — Winning Stage-2 candidate's cost/reliability, fixed vs. broken shots**

| Quantity | Case 1 ("fixed", n=3,164) | Case 3 ("broken", n=8,531) |
|---|---:|---:|
| `\|Delta_anc\|` mean / median | 3.49 / 2.74 | 5.21 / 2.74 |
| `\|Delta_anc\|` 90th pct. / max | 7.41 / 35.15 | 13.87 / 69.06 |
| Winning branch's relay iterations, mean / median | 746.9 / **50** | 3,881.8 / **1,486** |
| Winning branch's iterations <= 100 | 58.3% | 14.5% |
| Winning branch's iterations > 90% of 18,080-iter cap | 0.35% | 2.77% |
| Only 1 of 12 forced branches converged at all | 0.63% | **14.17%** |
| A 2nd-best forced branch also converged | 99.40% | 85.83% |
| Stage-1's own iteration count (for context), mean / median | 468.4 / 29.0 | 153.7 / 18.0 |
| Reported `forced_gap_ml` confidence value, mean / median | 2.89 / 2.44 | 4.12 / 1.94 |

The median weight advantage (`|Delta_anc|` median = 2.74 for both) is nearly
identical between the two populations — the *size* of the apparent weight gap does
not distinguish a trustworthy rescue from a spurious break. What does distinguish
them is *how the winning Stage-2 candidate was obtained*: "fixed" candidates
typically converge almost immediately (median 50 iterations, i.e. within the
initial `pre_iter=80` pass, before any relay-leg retry is even needed), whereas
"broken" candidates typically require ~1,486 iterations — roughly leg #23 of the
300 available relay legs (`(1486-80)/60 ~= 23.4`) — and are more than 20x as likely
to be the *only* one of the 12 forced branches that converged at all.

## 5. Discussion: Why RELAY-BP Reverses the Effect

This asymmetry corroborates an independent, earlier investigation in this
repository (`relay_too_slow/README.md`), conducted on a different operating point
(p=0.02, phenomenological noise) while diagnosing RELAY-BP's wall-clock cost for
`linearize_logicalgap`. That investigation found, at the level of raw
`decode_detailed()` calls:

- Stage-1 (natural syndrome): 100% convergence, mean 5.9 iterations.
- Stage-2 (forced/off-distribution syndrome): only 31.0% convergence; 69.0% of
  calls exhaust the full 18,080-iteration budget without ever converging. Among
  the calls that *do* converge, the mean cost is 3,243 iterations (~leg #54 of
  300).

The forced constraint pushes the decoding problem off the distribution BP is
implicitly tuned for (it asks the decoder to find a plausible correction in a
logical class the actual noise realization does not favor). RELAY-BP's relay
mechanism responds to this by retrying with independently perturbed memory
strengths (`gamma`) across up to `num_sets=300` legs, but with `stop_nconv=1` it
**accepts the very first leg that reaches a self-consistent fixed point** — it
never compares that leg's weight against alternative legs before returning. There
is therefore no guarantee, and only weak correlation, between "this Stage-2
instance converged" and "this Stage-2 instance's weight is close to the true
per-class minimum."

For an intrinsically rare event (a self-consistent BP fixed point on an
off-distribution, forced syndrome), searching through dozens of quasi-random
legs before one finally succeeds is itself an extreme-value selection process:
the leg that happens to converge is not a representative sample of "the weight of
the wrong class," but a survivorship-biased sample that can, by chance, land on an
anomalously *light* configuration purely because enough independent attempts were
made. Table 3's evidence is consistent with exactly this: "broken" wins are
disproportionately late-arriving (many legs consumed), disproportionately
uncorroborated (no second converged branch to sanity-check against), and yet
their raw weight is not distinguishably smaller than a "fixed" win's — they look
identical to the reselection rule, which cannot tell a genuine, well-supported
competing solution from a lucky one-shot fixed point.

BP-LSD does not exhibit this pathology because its post-processing (localized
statistics decoding, `LSD_CS`) is a systematic, deterministic refinement over BP's
belief vector rather than a stochastic multi-restart search: a smaller Stage-2
weight under BP-LSD is much more likely to reflect a genuinely competitive
alternative decoding, which is precisely why reselection is a net positive there
(Table 1, case 1 > case 3).

The `[[72,12,6]]` bivariate-bicycle code's large logical dimension (`K=12`)
compounds the effect purely combinatorially: with 12 independent forced branches
per shot instead of 1 (as in, e.g., a distance-only surface code), there are 12
independent chances per shot for RELAY-BP's survivorship-biased search to produce
one spuriously light wrong-class candidate.

## 6. Conclusion

1. The simulation and post-selection pipeline are implemented as intended; no
   code defect was found in `_forced_gap.py`, `_linearize_logicalgap.py`,
   `_constraints.py`, `result_collection.py`, `worker.py`, or
   `analysis/src/postselect.py`. The reversal is a real, reproducible effect
   in the data (bit-identical Stage-1 decodes were confirmed across independent
   runs).
2. The reversal is fully and exactly attributable to `count(case 3) - count(case
   1)`, i.e. correct-baseline-overridden-into-error events outnumbering
   wrong-baseline-rescued events (8,531 vs. 3,164 for RELAY-BP, the opposite
   ratio to BP-LSD's 10,285 vs. 11,664).
3. The mechanistic cause is RELAY-BP's `stop_nconv=1` policy: it accepts the
   first relay leg to converge on the (rare, off-distribution) forced sub-problem
   without cross-checking it against other legs, making the reported Stage-2
   weight for RELAY-BP an unreliable, high-variance estimate of the true
   per-class minimum weight — unlike BP-LSD, whose LSD post-processing gives a
   comparatively trustworthy weight estimate even under the same constraint.
4. Consequently, `forced_gap_ml`'s "adopt the global minimum weight" reselection
   rule is not decoder-agnostic in effectiveness: it is a net win only when the
   underlying decoder's weight comparisons across constrained sub-problems are
   themselves trustworthy. The anchored (`linearize_logicalgap`) strategy, which
   never trusts the Stage-2 weight for the final decision, is the safer choice
   for RELAY-BP under this configuration.

A natural (out of scope here, no code was modified for this report) follow-up
would be to test whether raising `stop_nconv` (requiring several legs to converge
before returning the best of them) reduces the case-3 rate for RELAY-BP, at the
cost of the already-substantial per-shot compute time.

## Appendix: Source Files Referenced

- `src/decoder_confidence/decoding/_forced_gap.py`
- `src/decoder_confidence/decoding/_linearize_logicalgap.py`
- `src/decoder_confidence/decoding/_constraints.py`
- `src/decoder_confidence/decoding/result_collection.py`
- `src/decoder_confidence/execution/worker.py`
- `analysis/src/postselect.py`
- `analysis/src/anchored_reselection.py`
- `documents/relay_bp_nonconvergence_behavior.md`
- `relay_too_slow/README.md`
- `analysis/circuit_level/circuit_level.ipynb` (cells 1-25, in particular the
  RELAY-BP post-selection curves in cells 24-25)
- Data: `simulation_result/code=bivariate_bicycle_code_Z,d=6,rounds=6,noisemodel=uniform,p=0.003,use_both=False,ibm_reproduce=True,xyz=False/decoding_result/`
  - `decoder=BP-LSD,metric=forced_gap_ml,get_all_failure_rate=True,get_detail_stat=True/`
  - `decoder=RELAY-BP,metric=forced_gap_ml,forced_unconverged_confidence_value='positive',get_detail_stat=True/`
  - `decoder=RELAY-BP,metric=linearize_logicalgap,forced_unconverged_confidence_value='positive',get_detail_stat=True/`
