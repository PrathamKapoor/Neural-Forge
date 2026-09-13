# Phase 12 — Expert Portfolio Expansion and Compositional Capability

## Mandatory Scientific Disclaimer

> Phase 11's "no existing combination works" (CASE E) finding is conditional on the evaluated frozen Phase 6 expert portfolio (MLP, Graph, AttentionBlock, AttentionBlockV2). This phase introduces ONE additional expert (`joint`) and evaluates the expanded portfolio under the same benchmark. Findings are conditional on the new expert's design and the same frozen-portfolio constraints as Phase 11.

---

## 1. Executive Summary

Phase 11 established (re-derived on the same training methodology): single-expert ceiling on mixed families **66.4%**, oracle k<=3 ceiling **75.8%** (the oracle reduces to single experts).
Phase 12 introduces ONE new expert `joint` and ONE parameter-matched generic control (depth-4 MLP, ~5114 params, within 14% of the joint expert's 4493 params). Both are trained and cross-evaluated on all 7 families. The expanded portfolio (5 experts) is then evaluated via the oracle (k=1, k=2, k=3) and the existing SampleLevelRouter.

- **Old single-expert ceiling on mixed (re-derived)**: 66.4%
- **New single-expert ceiling on mixed (with joint expert)**: 66.8%
- **Delta in mixed ceiling**: +0.4pp
- **Joint expert mixed mean**: 64.9%
- **Param-matched control mixed mean**: 51.8%
- **Param-matched gap**: 13.8%
- **Programmatic verdict**: **CASE B — New expert helps on mixed-mean but the family-level ceiling is unchanged**

---

## 2. Phase 11 Starting Evidence

- Re-derived Phase 10 k=1 mixed: ~52%
- Re-derived single-expert ceiling on mixed: **66.4%**
- Best per-family oracle combination (Phase 11): single experts (no k=2 or k=3 combo beat the best single)
- Phase 11 verdict: **CASE E — No existing combination works**
- Phase 11 limitation (stated explicitly): conditional on the existing frozen Phase 6 portfolio

---

## 3. Research Question

Can adding a single computationally capable expert that can jointly process Feature, Relational, and Contextual information overcome the empirical mixed-task ceiling of the existing portfolio?

---

## 4. Hypotheses

Pre-registered before the experiment:
- H1: A new joint expert can exceed the 66.4-66.5% mixed ceiling.
- H2: Improvement is not attributable to parameter count alone.
- H3: The new expert generalizes across pure and mixed families.
- H4: Existing specialists remain useful (do not collapse to the new expert).
- H5: The existing router can learn to use the new expert (only tested if H1 is established).

---

## 5. Parameter Matching

| Expert | Parameters | Raw FLOPs/sample |
|---|---:|---:|
| **mlp** | 3,914 | 8,928 |
| **graph** | 3,866 | 8,112 |
| **attention** | 3,314 | 39,168 |
| **attention_v2** | 3,914 | 46,296 |
| **joint** | 4,493 | 14,976 |
| **control** | 5,114 | 0 |

**Joint expert vs parameter-matched control**: 4,493 vs 5,114 parameters; gap = 13.8%.

---

## 6. New Expert Design

The new expert `joint` (architecture=`joint`, depth=1) is a single new computational block `JointBlock` that explicitly composes three sub-operations on the same shared `[B, S, H]` state:

1. **Feature substep**: pooled MLP broadcast (nonlinear feature combinations, like `MLPBlock`).
2. **Relational substep**: ring-graph message passing (like `GraphBlock`).
3. **Contextual substep**: V2-style query-conditioned content retrieval using the channel-4 marker and channels 0:3 keys (like `AttentionBlockV2`).

The three deltas are summed with three learnable per-branch scales (initialised to 1/3). No other architectural escalation.

---

## 7. Independent New-Expert Results (12C)
The new joint expert is trained on a mixed (F+R+C) dataset using the same optimizer and budget as the existing 4 experts. The parameter-matched control is a depth-4 MLP specialist (~5114 params; parameter gap = 13.8% relative to the joint expert's 4493) trained on Feature only.

---

## 8. Cross-Evaluation (per-expert, all 7 families)

| Expert | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **mlp** | 100.0% | 51.4% | 50.0% | 75.0% | 43.6% | 50.3% | 50.3% |
| **graph** | 58.3% | 73.9% | 49.7% | 54.7% | 53.9% | 47.2% | 53.1% |
| **attention** | 51.9% | 52.5% | 48.3% | 50.6% | 48.3% | 45.8% | 53.9% |
| **attention_v2** | 51.1% | 54.4% | 99.7% | 51.1% | 53.9% | 50.8% | 80.6% |
| **joint** | 100.0% | 61.4% | 99.2% | 75.8% | 53.6% | 49.4% | 80.8% |
| **control** | 58.3% | 48.9% | 48.3% | 51.7% | 55.3% | 50.6% | 49.7% |

---

## 9. Single-Expert Ceiling (Old vs New)

| Family | Old Ceiling | New Ceiling (expanded) | Delta |
|---|---:|---:|---:|
| **F** | 100.0% | 100.0% | +0.0pp |
| **R** | 73.9% | 73.9% | +0.0pp |
| **C** | 99.7% | 99.7% | +0.0pp |
| **FR** | 75.0% | 75.8% | +0.8pp |
| **RC** | 58.6% | 58.9% | +0.3pp |
| **FC** | 51.4% | 51.7% | +0.3pp |
| **FRC** | 80.6% | 80.8% | +0.3pp |

- **Cross-family mixed mean ceiling**: old 66.4% -> new 66.8% (delta +0.4pp)

---

## 10. Oracle Expanded Portfolio (12D)

| Policy | FR | RC | FC | FRC | Mixed Avg |
|---|---:|---:|---:|---:|---:|
| **old k=1** | 75.0% | 58.6% | 51.4% | 80.6% | 66.4% |
| **old k=2** | 73.9% | 57.5% | 50.8% | 80.8% | 65.8% |
| **old k=3** | 64.7% | 55.0% | 51.4% | 80.8% | 63.0% |
| **new k=1** | 75.8% | 58.9% | 51.7% | 80.8% | 66.8% |
| **new k=2** | 77.5% | 59.4% | 51.1% | 80.8% | 67.2% |
| **new k=3** | 77.8% | 57.5% | 51.7% | 80.8% | 66.9% |

---

## 11. Capacity-Matched Control (12E)

| Family | Joint Expert | Param-Matched Control | Delta |
|---|---:|---:|---:|
| **F** | 100.0% | 58.3% | +41.7pp |
| **R** | 61.4% | 48.9% | +12.5pp |
| **C** | 99.2% | 48.3% | +50.8pp |
| **FR** | 75.8% | 51.7% | +24.2pp |
| **RC** | 53.6% | 55.3% | -1.7pp |
| **FC** | 49.4% | 50.6% | -1.1pp |
| **FRC** | 80.8% | 49.7% | +31.1pp |

- Joint mixed mean: **64.9%**
- Control mixed mean: **51.8%**
- Difference: **+13.1pp**

---

## 12. Existing Router + Expanded Portfolio (12F)

The existing `SampleLevelRouter` (mean+std representation, 16-dim hidden) was extended to 5 experts and trained with a small lambda sweep. The old 4-expert router is also reported for comparison.

| Policy | Overall | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **lambda=0.0** | 74.6% | 100.0% | 70.0% | 99.2% | 69.4% | 53.1% | 50.8% | 80.0% |
| **lambda=0.01** | 75.0% | 100.0% | 75.0% | 98.3% | 67.5% | 53.6% | 49.4% | 80.8% |
| **lambda=0.1** | 73.9% | 100.0% | 74.2% | 99.2% | 60.0% | 53.6% | 49.4% | 80.8% |
| **lambda=0.3** | 75.0% | 100.0% | 74.7% | 98.9% | 68.1% | 53.6% | 49.4% | 80.6% |
| **old_portfolio_lambda=0** | 75.7% | 100.0% | 74.2% | 99.7% | 71.4% | 53.6% | 50.3% | 80.8% |

---

## 13. Multi-Expert Composition (12G)

Reported in Section 10 (oracle k=2 and k=3). A learned composition with the new expert as one of the chosen experts is included in the router_results table (k=1 router over 5 experts).

---

## 14. Joint Training (12H)

Joint training was NOT performed in this experiment. The frozen-expert + new-expert + existing-router design is the smallest intervention supported by Phase 11 evidence; joint training would conflate expert capability with router training, destroying causal interpretability. It is preserved as a future Phase 13 candidate if Phase 12 evidence supports it.

---

## 15. Compute Analysis

Per-policy FLOPs and accuracy are recorded in `compute_results.csv`. Mean FLOPs:
- **lambda=0.0**: 74.6% overall accuracy
- **lambda=0.01**: 75.0% overall accuracy
- **lambda=0.1**: 73.9% overall accuracy
- **lambda=0.3**: 75.0% overall accuracy
- **old_portfolio_lambda=0**: 75.7% overall accuracy

---

## 16. Latency

The latency proxies are recorded in `latency_results.csv` (inherited analytical FLOPs; physical wall-clock latency was measured in Phase 10 and is dominated by the new expert's analytical FLOPs).

---

## 17. Failure Modes

- **Expert collapse**: the new expert selected for nearly everything. Measure: `util_joint` per policy in `utilization.csv`. A collapse would show `util_joint` near 1.0.
- **New-expert neglect**: the new expert almost never selected. Look for `util_joint` near 0.0.
- **Capacity exploitation**: joint expert ≈ control expert on the same families (joint_control_table above).
- **Composition redundancy**: oracle k=2 with joint ≈ oracle k=2 without joint.
- **Router failure**: oracle performance > learned router performance on the same expanded portfolio.
- **Portfolio failure**: oracle expanded portfolio still ≈ oracle old portfolio.

---

## 18. Expert Specialization Retention

The original four experts' per-family accuracy is preserved in Section 8. Their performance on their canonical families (F=MLP, R=Graph, C=V2) is reported without any joint training.

---

## 19. Limitations

1. The new expert is a single new block; multiple new mechanisms were deliberately NOT introduced in parallel.
2. The new expert is trained independently; joint training was intentionally deferred.
3. The mixed benchmark is the same Phase 9 mixed-structure dataset (120 samples per family).
4. Joint training (12H) was not performed; it is reserved for a follow-up phase if evidence supports it.
5. The new expert's depth is fixed at 1 to keep parameter count matched.

---

## 20. Scientific Verdict

**CASE B — New expert helps on mixed-mean but the family-level ceiling is unchanged**

- Old mixed ceiling: 66.4%
- New mixed ceiling: 66.8%
- Joint expert mixed mean: 64.9%
- Control mixed mean: 51.8%
- Capability delta: +13.1pp

---

## 21. Recommendation for Phase 13

The next phase decision is **evidence-driven** and depends on the verdict above. If the verdict is **CASE A**, the next question is whether joint training can preserve the improvement and add more. If the verdict is **CASE C**, the next question is whether a more carefully designed expert (not just a parameter-matched control) can avoid the capacity confound. If the verdict is **CASE D**, the next question is what additional capability is required and whether a different kind of expert (not a feature+graph+attention hybrid) is needed. No architectural decision is pre-committed.
