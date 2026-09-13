# Phase 19 — Mixed-Benchmark Construction Repair and Composition Revalidation

## Mandatory scientific correction

> Historical mixed-composition results are retained but are NOT treated as valid
> evidence for claims requiring an observable relational component on RC/FRC.

## 1. What was wrong / why it mattered / what was (in)validated

- Wrong: the relational carrier was inserted (ch0) before the contextual overwrite (ch0:3).
- Mattered: labels kept depending on the erased statistic `sr` on RC/FRC.
- Invalidated: any RC/FRC interpretation requiring recovery of the erased relational
  information (this includes the Phase 13–17 composition-failure narrative for RC).
- Remains valid: standalone R experiments and inputs where the signal stayed observable,
  subject to their own controls.
- Repaired: ONLY the benchmark construction. The architecture was NOT repaired.

## 2. Repair (minimal, versioned `phase19-repaired-v1`)

Base → Contextual (identical writes) → Feature offsets (common-mode, survive) →
Relational carrier on channel 5 (previously pure noise; provably survives).
Same RNG values, same label algebra, same families/balance/sequences, still [B,S,8].

## 3. Bug reproduction (19A, original construction)

- R-align on R: 97.8% (observable)
- R-align on RC: 48.1% (erased)
- R-align on FRC: 55.6% (erased)

## 4. Observability (19C, repaired)

- R-align R/RC/FRC/FR: 98.9% / 97.2% / 98.3% / 98.6%
- Full per-family carrier table in `observability.csv` + `dependency_audit.csv`.

## 5. Gates G1–G8

| Gate | Passed | Detail |
|---|---|---|
| **G1_observability** | True | all component carriers observable |
| **G2_no_leakage** | True | max single-channel label probe 0.744 (bar 0.9) |
| **G3_no_family_leakage** | True | family-probe delta repaired−original -0.005 (tol 0.05) |
| **G4_invariance** | True | C retrieval perm drop +0.000; R-family markvar delta repaired -0.158 vs original -0.183 (RC-family deltas are floor effects, reported in CSV); perm R-drop repaired +0.483 vs original +0.467 |
| **G5_counterfactual** | True | R-valid 0.969, C-valid 0.972 (bar 0.9) |
| **G6_semantics** | True | agree 61/120, disagree 59; FRC ties absent: True |
| **G7_learnability** | True | rule_RC 0.967; lagprobe_R 0.895; mlp R/RC/FRC [0.467, 0.483, 0.733] (context); stat_R 0.975 |
| **G8_reproducibility** | True | exact tensor equality across rebuilds |

## 6. Counterfactual validity (19F) and semantics (19E)

- R/C counterfactual valid rates in `counterfactual_validation.csv`.
- Rule (statistic+retrieval) RC: 97.8% — RC genuinely requires both observable components iff H7 passes.

## 7. Hypotheses H1–H8 (programmatic)

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1** | SUPPORTED | Original: R-align R 97.8% vs RC 48.1%/FRC 55.6%: erasure reproduced. |
| **H2** | SUPPORTED | Repaired R-align on RC 97.2% (bar 90%). |
| **H3** | SUPPORTED | Repaired R-align on FRC 98.3% (bar 90%). |
| **H4** | SUPPORTED | All mixed-task components observable. |
| **H5** | SUPPORTED | C retrieval perm drop +0.000; R-family markvar delta repaired -0.158 vs original -0.183 (RC-family deltas are floor effects, reported in CSV); perm R-drop repaired +0.483 vs original +0.467 |
| **H6** | SUPPORTED | max single-channel label probe 0.744; family delta -0.005. |
| **H7** | SUPPORTED | Statistic+retrieval rule RC 97.8% vs C-only 48.3%: RC genuinely requires both. |
| **H8** | SUPPORTED | All validity gates pass. |

## 8. Final CASE (programmatic)

**CASE F — Benchmark validity fully established and composition track can safely reopen**

Recommendation: Reopen composition with a clean portfolio re-evaluation first; no new architecture yet.

## 9. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
| bug_not_reproduced | False | original RC/FRC R-align < 60% with R ≥ 90% | RC 0.481, FRC 0.556 |
| carrier_unobservable_RC | False | repaired R-align on RC < 90% | 0.972 |
| carrier_unobservable_FRC | False | repaired R-align on FRC < 90% | 0.983 |
| gate_G1_observability_failed | False | G1_observability must pass | all component carriers observable |
| gate_G2_no_leakage_failed | False | G2_no_leakage must pass | max single-channel label probe 0.744 (bar 0.9) |
| gate_G3_no_family_leakage_failed | False | G3_no_family_leakage must pass | family-probe delta repaired−original -0.005 (tol 0.05) |
| gate_G4_invariance_failed | False | G4_invariance must pass | C retrieval perm drop +0.000; R-family markvar delta repaired -0.158 vs original -0.183 (RC-family deltas are floor effects, reported in CSV); perm R-drop repaired +0.483 vs original +0.467 |
| gate_G5_counterfactual_failed | False | G5_counterfactual must pass | R-valid 0.969, C-valid 0.972 (bar 0.9) |
| gate_G6_semantics_failed | False | G6_semantics must pass | agree 61/120, disagree 59; FRC ties absent: True |
| gate_G7_learnability_failed | False | G7_learnability must pass | rule_RC 0.967; lagprobe_R 0.895; mlp R/RC/FRC [0.467, 0.483, 0.733] (context); stat_R 0.975 |
| gate_G8_reproducibility_failed | False | G8_reproducibility must pass | exact tensor equality across rebuilds |

## 10. Reopen condition (§25)

Composition reopens ONLY on CASE F (full gate pass), and then with a clean
portfolio re-evaluation first — no new architecture yet.

## 11. Limitations

1. Validity is construction-level; model behavior on the repaired benchmark is a future phase.
2. FR raw-statistic caveat (F-offset dominance) is handled by demeaned/channel-appropriate statistics, documented in the audit.
3. Three seeds; deterministic generation verified by exact rebuild equality.
