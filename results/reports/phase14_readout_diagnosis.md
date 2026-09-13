# Phase 14 — Relational Readout, Fusion, and Prediction-Head Bottleneck Diagnosis

## Mandatory Scientific Disclaimer

> Phase 13 established that the JointCo representation contains decodable relational information (R-signal probe = 76.9% on R) but the JointCo's actual R accuracy is 59.7%. Phase 14 investigates where, between the relational representation and the final prediction, the information is being lost. All primary diagnostic conditions use a FROZEN JointCo encoder and a trainable small diagnostic head. The JointCo expert is not jointly retrained.

---

## 1. Executive Summary

- **Phase 14 baseline (Phase 13 JointCo reproduction)**: mixed mean = 65.0%; R = 62.2%, RC = 53.3%, FRC = 80.3%
- **R-signal probe on the frozen representation (Phase 13 re-derivation)**: R = 77.2%
- **Best pooling (P1-P4) on R**: 61.4%
- **Best head (linear vs small nonlinear) on R**: 55.3%
- **Direct relational-branch readout (rel_only head) on R**: 48.6%
- **Old single-expert ceiling (mixed)**: 66.3%
- **New single-expert ceiling (mixed, with original Phase 13 head)**: 67.2%
- **New single-expert ceiling (mixed, with small_nonlinear head substituted on jointco)**: 67.4%
- **Programmatic verdict**: **CASE D — Relational branch itself requires improvement**

---

## 2. Phase 13 Starting Evidence

| Family | Phase 13 JointCo |
|---|---:|
| F | 100.0% |
| R | 61.4% |
| C | 99.2% |
| FR | 75.8% |
| RC | 53.6% |
| FC | 49.4% |
| FRC | 80.8% |

R-signal probe (Phase 13): 76.9%. Phase 13 verdict: CASE D (fusion is the bottleneck; R info present but unused).

---

## 3. Research Question

> Where between the relational representation and the final prediction does useful relational information become unusable?

---

## 4. Hypotheses

- H1: Readout replacement improves relational prediction.
- H2: Improvement is specifically relational.
- H3: Pooling contributes to the bottleneck.
- H4: Relational branch output is more useful than the final fused output.
- H5: Improvement is not caused by increased representation capacity.

---

## 5. Primary Comparison Table (25)

| Representation | Pooling | Head | F | R | C | FR | RC | FC | FRC | Pure | Mixed | Overall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Phase 13 baseline (original head, query pool)** | 100.0% | 62.2% | 99.4% | 76.4% | 53.3% | 50.0% | 80.3% | 87.2% | 65.0% | 74.5% |
| **Frozen encoder + P1_query + linear head** | 91.7% | 55.3% | 98.1% | 71.9% | 53.6% | 49.7% | 77.2% | 71.1% |
| **Frozen encoder + P1_query + small_nonlinear head** | 96.7% | 54.7% | 95.8% | 74.4% | 53.3% | 47.5% | 76.9% | 71.3% |
| **Frozen encoder + P2_mean + linear head** | 100.0% | 56.1% | 88.6% | 76.4% | 51.4% | 50.0% | 69.7% | 70.3% |
| **Frozen encoder + P3_max + linear head** | 98.3% | 59.7% | 91.1% | 76.9% | 51.7% | 51.1% | 73.9% | 71.8% |
| **Frozen encoder + P4_mean_plus_query + linear head** | 91.7% | 57.8% | 96.9% | 74.2% | 53.6% | 50.6% | 77.5% | 71.7% |
| **Frozen encoder + rel_only branch + linear head** | 56.1% | 48.6% | 47.2% | 71.9% | 56.7% | 49.2% | 47.2% | 53.8% |
| **Frozen encoder + feat_only branch + linear head** | 66.7% | 51.1% | 47.2% | 59.4% | 49.7% | 47.8% | 48.3% | 52.9% |
| **Frozen encoder + ctx_only branch + linear head** | 50.0% | 55.6% | 99.4% | 35.3% | 53.6% | 51.1% | 79.2% | 60.6% |
| **Frozen encoder + branches_concat + linear head** | 100.0% | 56.7% | 93.9% | 70.6% | 53.1% | 50.8% | 76.4% | 71.6% |
| **Frozen encoder + fused_delta + linear head** | 58.3% | 60.3% | 94.2% | 53.3% | 53.9% | 50.0% | 75.3% | 63.6% |
| **Frozen encoder + scaled_sum + linear head** | 75.0% | 54.4% | 85.3% | 68.3% | 51.9% | 49.7% | 69.7% | 64.9% |
| **Frozen encoder + block_output + linear head** | 93.6% | 55.6% | 85.3% | 72.2% | 52.8% | 48.1% | 71.9% | 68.5% |

---

## 6. Causal Diagnosis Table (26)

| Candidate Bottleneck | Status | Evidence (seed 0) |
|---|---|---|
| **POOLING** | NOT SUPPORTED | Changing pooling alone does not improve R (-3.3pp) or RC (+0.0pp). |
| **CLASSIFIER_HEAD** | NOT SUPPORTED | Nonlinear head does not improve over linear (R -2.5pp, RC +0.8pp). |
| **FUSION** | NOT SUPPORTED | Fused-delta head (R=60.0%) is at least as good as branch-concat head (R=55.0%); no fusion loss. |
| **RELATIONAL_BRANCH** | NOT SUPPORTED | Direct relational-branch readout (mean 51.4%) does not exceed baseline head (mean 63.6%); the relational branch alone is not task-sufficient. |
| **REPRESENTATION_TRANSFER** | SUPPORTED | R-signal probe = 78.3% but baseline R final accuracy = 59.2%; the existing head fails to use the information. |
| **BENCHMARK** | NOT SUPPORTED | At least one diagnostic intervention improves R by > 2pp over baseline (best R = 63.3% vs 59.2%). |

---

## 7. Branch Combination Analysis (16)

| Combination | F | R | C | FR | RC | FC | FRC | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **feat** | 100.0% | 59.2% | 54.2% | 76.7% | 53.6% | 49.7% | 50.3% | 63.4% |
| **rel** | 50.0% | 54.4% | 49.7% | 72.5% | 50.8% | 47.8% | 47.8% | 53.3% |
| **ctx** | 50.0% | 51.4% | 99.7% | 43.1% | 53.1% | 50.3% | 79.2% | 61.0% |
| **feat+rel** | 100.0% | 57.2% | 51.9% | 76.1% | 53.3% | 48.6% | 46.1% | 61.9% |
| **feat+ctx** | 100.0% | 57.2% | 96.4% | 76.4% | 53.6% | 50.6% | 78.9% | 73.3% |
| **rel+ctx** | 50.0% | 57.2% | 96.7% | 72.2% | 54.2% | 50.3% | 78.6% | 65.6% |
| **feat+rel+ctx** | 100.0% | 59.4% | 94.7% | 75.6% | 52.8% | 48.9% | 77.8% | 72.7% |

R+C vs C (increment of R on top of C): +5.8pp.
R+C vs R (increment of C on top of R): +46.9pp.

---

## 8. Relational Destruction (21)

Best-pool + small-nonlinear head evaluated on original vs relationally-permuted features:

| Condition | R | RC | FRC |
|---|---:|---:|---:|
| **original** | 54.7% | 53.3% | 76.9% |
| **relational_permuted** | 55.6% | 52.5% | 77.5% |

R destruction drop: -0.8pp.

---

## 9. Latency Decomposition (24)

| Stage | Mean (us) |
|---|---:|
| **encoder_only** | 27.2 |
| **encoder_plus_mean_pool** | 28.7 |
| **encoder_plus_query_pool** | 32.0 |
| **linear_head_total** | 29.3 |
| **small_nonlinear_head_total** | 33.4 |

---

## 10. Final Bottleneck Diagnosis (12)

Primary finding: CASE D — Relational branch itself requires improvement.

---

## 11. Limitations

1. Only the frozen-encoder diagnostic protocol is used in the primary experiment. Joint retraining is not performed.
2. Only one new head architecture (small nonlinear) and three alternative pooling strategies are tested.
3. The 3-seed sample is small.
4. The head_epochs and jointco_epochs are predeclared constants; no additional tuning is performed.

---

## 12. Decision for Phase 15

Phase 15 must be determined from the verdict. Possible directions:

- If the head is the bottleneck: try a slightly larger nonlinear head (24 -> 48 -> 2) or a head that explicitly reads the relational branch.
- If pooling is the bottleneck: implement a small learned attention pooling (one attention head) — but only if 14C supports it.
- If fusion is the bottleneck: re-architect the JointCo fusion (e.g., 2-step fusion).
- If the relational branch itself is insufficient: this is the spec's signal to consider a fundamentally new expert or a much deeper relational substep. That is a major architectural move and must be motivated by clear evidence.

No architectural decision is pre-committed.
