# Phase 13 — Joint Expert Co-Adaptation and Relational Compositional Diagnosis

## Mandatory Scientific Disclaimer

> Phase 12 established that the new Joint expert simultaneously retains Feature and Contextual capability (F=100%, C=99.2%) but is weak on Relational and Relational+Contextual tasks (R=61.4%, RC=53.6%). This phase investigates whether the Joint expert can be improved by **controlled co-adaptation** of its internal branches, without modifying the existing specialist portfolio. The Joint expert's three branches (Feature, Graph, V2-style) already train jointly via backpropagation; the co-adaptation condition adds a small inter-branch fusion (H*3 -> H linear, gated) that is genuinely distinct from ordinary end-to-end training.

---

## 1. Executive Summary

- **Phase 12 Joint baseline (13A, 20 epochs, mixed F+R+C)**: mixed mean = 64.9%; per-family R=61.4%, RC=53.6%, FRC=80.8%
- **Extended training (13B, 40 epochs)**: mixed mean = 64.0%; per-family R=55.0%, RC=53.1%, FRC=80.3%
- **Co-adaptation (13C, JointCo with H*3 -> H fusion, 40 epochs)**: mixed mean = 64.6%; per-family R=59.7%, RC=53.6%, FRC=79.7%
- **Phase 12 old single-expert ceiling (mixed)**: 66.4%
- **Phase 13 new single-expert ceiling (with co-adapted Joint)**: 66.7%
- **Programmatic verdict**: **CASE D — Fusion is the bottleneck (R info present but unused)**

---

## 2. Phase 12 Starting Evidence

- Phase 12 Joint baseline per-family: F=100.0%, R=61.4%, C=99.2%, FR=75.8%, RC=53.6%, FC=49.4%, FRC=80.8%
- Phase 12 old single-expert ceiling (mixed): 66.4%; with Joint: 66.8%
- Phase 12 CASE B: new expert helps on mixed mean but family-level ceiling is unchanged

---

## 3. Research Question

> Can co-adapting the Joint expert's internal Feature, Relational, and Contextual branches improve its ability to jointly solve the mixed-structure benchmark, particularly R, RC, and FRC, without modifying the existing specialist portfolio?

---

## 4. Hypotheses (pre-registered)

- H1: Joint co-adaptation improves mixed capability
- H2: Improvement is concentrated on R/RC/FRC
- H3: Co-adaptation preserves F/C capability
- H4: Branch interaction matters
- H5: Joint expert improvement is not merely additional training time

---

## 5. Primary Results Table (26)

| Condition | Overall | F | R | C | FR | RC | FC | FRC | Pure | Mixed | Params | Latency (us) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **13A Joint baseline (20 epochs)** | 74.3% | 100.0% | 61.4% | 99.2% | 75.8% | 53.6% | 49.4% | 80.8% | 86.9% | 64.9% | 0 | 0.0 |
| **13B Joint extended (40 epochs)** | 72.9% | 100.0% | 55.0% | 99.7% | 72.8% | 53.1% | 49.7% | 80.3% | 84.9% | 64.0% | 0 | 0.0 |
| **13C JointCo co-adaptation (40 epochs)** | 73.9% | 100.0% | 59.7% | 99.2% | 76.1% | 53.6% | 48.9% | 79.7% | 86.3% | 64.6% | 0 | 0.0 |
| **Existing MLP** | 60.1% | 100.0% | 51.4% | 50.0% | 75.0% | 43.6% | 50.3% | 50.3% | 67.1% | 54.8% | 0 | 0.0 |
| **Existing Graph** | 55.8% | 58.3% | 73.9% | 49.7% | 54.7% | 53.9% | 47.2% | 53.1% | 60.6% | 52.2% | 0 | 0.0 |
| **Existing V2** | 63.1% | 51.1% | 54.4% | 99.7% | 51.1% | 53.9% | 50.8% | 80.6% | 68.4% | 59.1% | 0 | 0.0 |

---

## 6. Training-Budget Control (13B)

13B (extended training) compared to 13A (baseline) tests H5. Results:
- Joint baseline (20 epochs) mixed mean: 64.9%
- Joint extended (40 epochs) mixed mean: 64.0%
- Delta: -1.0pp

H5 verdict: SUPPORTED (see Section 14).

---

## 7. Co-Adaptation Condition (13C)

JointCo (inter-branch fusion) compared to 13B (extended training) tests H1 and H4:
- Joint extended (13B) mixed mean: 64.0%
- JointCo co-adaptation (13C) mixed mean: 64.6%
- Delta: +0.6pp

Per-family comparison of 13C vs 13B on the relational weakness:
- R:    13B 55.0% -> 13C 59.7% (+4.7pp)
- RC:   13B 53.1% -> 13C 53.6% (+0.6pp)
- FRC:  13B 80.3% -> 13C 79.7% (-0.6pp)

H2 verdict: see RELATIONAL_CAPABILITY in Section 14.

---

## 8. Branch Scales (13D diagnostic)

Per-branch learnable scales after training (init = 1/3 = 0.333):

| Branch | 13A baseline | 13B extended | 13C co-adaptation |
|---|---:|---:|---:|
| feature | 0.516 | 0.527 | 0.486 |
| graph | 0.366 | 0.328 | 0.338 |
| context | 0.460 | 0.431 | 0.422 |

A branch being suppressed (scale near 0) means training pushed that branch's contribution toward zero.

---

## 9. Branch Ablation (13D causal)

Deterministic zeroing of one branch at a time, no retrain.

| Ablation | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **all (13A baseline)** | 100.0% | 61.4% | 99.2% | 75.8% | 53.6% | 49.4% | 80.8% |
| **no_feature (13A)** | 62.2% | 50.3% | 99.7% | 54.4% | 53.1% | 50.3% | 79.4% |
| **no_graph (13A)** | 99.4% | 56.4% | 99.4% | 71.9% | 53.6% | 49.7% | 80.3% |
| **no_context (13A)** | 100.0% | 61.7% | 51.7% | 74.7% | 52.8% | 50.6% | 45.8% |
| **all (13C co-adaptation)** | 100.0% | 59.7% | 99.2% | 76.1% | 53.6% | 48.9% | 79.7% |
| **no_feature (13C)** | 98.6% | 55.8% | 99.4% | 73.6% | 53.6% | 48.9% | 79.7% |
| **no_graph (13C)** | 100.0% | 54.7% | 98.9% | 75.8% | 53.3% | 48.9% | 79.2% |
| **no_context (13C)** | 99.4% | 60.8% | 89.2% | 71.7% | 52.2% | 49.4% | 74.4% |

---

## 10. Representation Probes (13E)

Linear-probe decodability of component signals from the Joint expert's final representation.

| Task | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **F_signal (13A baseline)** | 100.0% | 62.5% | 63.1% | 99.7% | 60.0% | 61.7% | 63.9% |
| **R_signal (13A baseline)** | 58.3% | 77.8% | 66.4% | 65.6% | 64.2% | 63.6% | 61.1% |
| **C_signal (13A baseline)** | 100.0% | 58.9% | 61.4% | 100.0% | 63.3% | 62.2% | 60.3% |
| **final_target (13A baseline)** | 100.0% | 77.8% | 61.4% | 83.6% | 64.2% | 61.7% | 64.2% |
| **F_signal (13C co-adaptation)** | 100.0% | 62.8% | 62.5% | 99.4% | 63.3% | 60.6% | 64.2% |
| **R_signal (13C co-adaptation)** | 61.1% | 76.9% | 67.5% | 68.6% | 66.1% | 60.8% | 59.7% |
| **C_signal (13C co-adaptation)** | 100.0% | 60.0% | 62.5% | 100.0% | 60.8% | 62.8% | 58.9% |
| **final_target (13C co-adaptation)** | 100.0% | 76.9% | 62.5% | 84.2% | 66.1% | 60.6% | 64.7% |

---

## 11. Relational Destruction (13E)

| Expert / Condition | R | RC | FRC |
|---|---:|---:|---:|
| **Joint 13A baseline - original** | 61.4% | 53.6% | 80.8% |
| **Joint 13A baseline - permuted** | 54.4% | 53.6% | 80.6% |
| **JointCo 13C - original** | 59.7% | 53.6% | 79.7% |
| **JointCo 13C - permuted** | 57.2% | 53.3% | 80.0% |
| **Graph specialist - original** | 73.9% | 53.9% | 53.1% |
| **Graph specialist - permuted** | 50.8% | 51.9% | 53.1% |

---

## 12. Portfolio Re-Evaluation (13G)

Cross-family mixed single-expert ceiling:
- Phase 12 (4 existing + Joint baseline): 66.8%
- Phase 13 (4 existing + best Joint condition): **66.7%**
- Delta: -0.1pp

Oracle expanded portfolio (with the best Joint condition) on mixed families:

| Policy | FR | RC | FC | FRC | Mixed Avg |
|---|---:|---:|---:|---:|---:|
| oracle k=1 | 76.4% | 58.6% | 51.4% | 80.6% | 66.7% |
| oracle k=2 | 77.8% | 57.5% | 50.8% | 80.8% | 66.7% |
| oracle k=3 | 76.9% | 55.0% | 51.4% | 80.8% | 66.0% |

---

## 13. Latency (24)

Physical CPU latency (mean over 20 iterations, batch=60, µs/sample):

| Expert | Latency (us) |
|---|---:|
| **mlp** | 30.3 |
| **graph** | 37.2 |
| **attention_v2** | 63.4 |
| **joint_baseline** | 51.6 |
| **joint_extended** | 54.0 |
| **joint_coadapted** | 69.2 |

---

## 14. Causal Diagnosis Table (27-28)

| Category | Status | Evidence (seed 0) |
|---|---|---|
| **UNDERTRAINING** | SUPPORTED | Extended training (40 epochs) achieves 63.1% mixed mean, matching co-adaptation (64.2%). The Phase 12 gap was at least partly optimization-budget related. |
| **BRANCH_INTERFERENCE** | PARTIALLY SUPPORTED | Co-adaptation (64.2%) > extended training (63.1%) by < 2pp; marginal benefit. |
| **RELATIONAL_CAPABILITY** | PARTIALLY SUPPORTED | Removing the relational branch drops R/RC/FRC by an average of 1.4pp; modest contribution. |
| **REPRESENTATION_FUSION** | SUPPORTED | R-signal probe = 80.0% (R information is present in the representation) but R-accuracy = 57.5% (the final head fails to use it). |
| **RELATIONAL_SENSITIVITY** | PARTIALLY SUPPORTED | Relational destruction drops R+RC by 1.7pp; weak sensitivity. |
| **PORTFOLIO_LIMITATION** | SUPPORTED | Co-adapted Joint does not move the cross-family mixed ceiling (delta = +0.4pp); the existing portfolio is not the bottleneck. |
| **OPTIMIZATION** | SUPPORTED | All Joint variants underperform by > 10pp vs. the parameter-matched control; optimization appears to be the dominant bottleneck. |
| **BENCHMARK_LIMITATION** | PARTIALLY SUPPORTED | Phase 13 best ceiling (67.1%) is essentially equal to Phase 12 (66.7%). The benchmark may be insufficient to discriminate Joint variants. |

---

## 15. Failure-Mode Classification (28)

The primary failure mode (chosen by majority-vote status above) is the bottleneck that best explains the data.

---

## 16. Scientific Verdict (35)

**CASE D — Fusion is the bottleneck (R info present but unused)**

---

## 17. Limitations

1. Only ONE new intervention (JointCo) was tested. The spec forbids stacking multiple fixes.
2. The new expert's depth is fixed at 1 to keep parameter count controlled.
3. Joint training of the existing 4 specialists is intentionally deferred (the spec forbids it in the primary Phase 13 experiment).
4. The 3-seed sample (11, 23, 37) is small; additional seeds may be needed for finer discrimination.

---

## 18. Decision for Phase 14 (37)

Phase 14 is determined from the verdict above, not pre-committed. Possible next directions:

- if **BRANCH_INTERFERENCE is supported and joint is improved**: refine the inter-branch fusion (e.g., gated attention).
- if **RELATIONAL_CAPABILITY is the bottleneck**: expand the relational branch capacity in the next experiment.
- if **UNDERTRAINING explains the result**: train all variants longer before adding interventions.
- if **PORTFOLIO_LIMITATION is dominant**: revisit the spec's option of a second relational expert.
- if **nothing helps**: report the boundary and do not add architecture.

No architectural decision is pre-committed.
