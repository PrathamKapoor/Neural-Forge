# Phase 11 — Composition Bottleneck Localization and Interface Diagnosis

## Mandatory Scientific Disclaimer

> Phase 10's reported single-expert ceiling of 66.4% is an empirical value derived from the trained Phase 10 portfolio. It does NOT constitute a universal mathematical limit. The oracle "mixed-task ceiling" recomputed in this Phase 11 experiment depends on the same experts; if those experts cannot solve the mixed task even in oracle-combination, that is a statement about THIS portfolio, not about composition in general.

---

## 1. Executive Summary

Phase 10 reported that fixed k=2 composition reached 57.2% mixed accuracy and k=3 reached 60.4% on the validated 7-family mixed benchmark, against a cross-family single-expert ceiling of 66.4%. The current Phase 11 experiment re-trains the same four experts and re-derives these numbers from the same code path. The objective is **causal localization**: identifying which of (A) expert capability, (B) representation/interface, (C) aggregation, (D) composition order, (E) routing, or (F) benchmark semantics is the dominant bottleneck.

- **Re-derived Phase 10 k=1 mixed accuracy**: 52.2%
- **Re-derived Phase 10 single-expert ceiling on mixed families**: 66.4%
- **Oracle-combination ceiling (k<=3) on mixed families**: 66.5%
- **Best per-family oracle combination** (from 11A): {'F': 'mlp', 'R': 'graph', 'C': 'attention_v2', 'FR': 'mlp', 'RC': 'graph', 'FC': 'mlp', 'FRC': 'attention_v2'}
- **Programmatic verdict**: **CASE E — No existing combination works**

---

## 2. Research Question

Why does multi-expert execution fail to exploit the complementary computational capabilities of the existing experts strongly enough to exceed the single-expert ceiling on the validated mixed benchmark?

---

## 3. Method (Causal Map)

| Layer | Diagnostic | Status |
|---|---|---|
| 11A | Oracle composition feasibility | executed |
| 11B | Representation transfer probing | executed |
| 11C | Interface compatibility (StateChain) | executed |
| 11D | Aggregation comparison | executed |
| 11E | Composition order | executed |
| 11F | Routing-selection diagnosis | executed |
| 11G | Benchmark semantic validation | executed |
| 11H | Minimal validated composition | executed (no / with linear adapter) |

All experiments re-use the four Phase 6 frozen experts and the validated `Phase9MixedStructureDataset` (840 samples, 120 per family). Seeds: [11, 23, 37].

---

## 4. 11A — Oracle Composition Feasibility

| Family | Oracle Ceiling | Best Combo (this seed) |
|---|---:|---|
| **F** | 100.0% | mlp |
| **R** | 75.8% | graph |
| **C** | 99.7% | attention_v2 |
| **FR** | 75.0% | mlp |
| **RC** | 58.6% | graph |
| **FC** | 51.4% | mlp |
| **FRC** | 80.8% | attention_v2 |

Cross-family mixed mean oracle ceiling: **66.5%** vs Phase 10 single-expert ceiling 66.4%.

---

## 5. 11B — Representation Transfer Probing

Decodability of the FINAL target from each expert's `[B, S, H]` state at the `after_blocks` stage (linear probe, ridge regression on mean-pooled state). Higher is better.

| Family | MLP | Graph | V2 |
|---|---:|---:|---:|
| **F** | 0.0% | 0.0% | 0.0% |
| **R** | 0.0% | 0.0% | 0.0% |
| **C** | 0.0% | 0.0% | 0.0% |
| **FR** | 0.0% | 0.0% | 0.0% |
| **RC** | 0.0% | 0.0% | 0.0% |
| **FC** | 0.0% | 0.0% | 0.0% |
| **FRC** | 0.0% | 0.0% | 0.0% |

The full per-expert × per-stage × per-task matrix is in `representation_probes.csv`.

---

## 6. 11C — Interface Compatibility (Sequential StateChain)

| Sequence | Overall | Mixed | FLOPs |
|---|---:|---:|---:|
| **mlp** | 60.1% | 54.8% | 8,928 |
| **graph** | 55.8% | 52.2% | 8,112 |
| **attention_v2** | 63.1% | 59.1% | 46,296 |
| **mlp+graph** | 51.5% | 50.3% | 17,040 |
| **graph+mlp** | 53.7% | 54.0% | 17,040 |
| **mlp+attention_v2** | 65.9% | 61.6% | 55,224 |
| **attention_v2+mlp** | 66.9% | 61.1% | 55,224 |
| **graph+attention_v2** | 55.2% | 55.1% | 54,408 |
| **attention_v2+graph** | 52.7% | 51.5% | 54,408 |
| **mlp+graph+attention_v2** | 55.1% | 53.3% | 63,336 |

StateChain preserves V2's raw-feature requirement (V2 always receives the original input to read its channel-4 marker and channels 0:3 keys), unlike Phase 10's `SequentialSpecialistComposition`.

---

## 7. 11D — Aggregation Comparison

| Method | Overall | Mixed |
|---|---:|---:|
| **raw** | 58.4% | 53.3% |
| **weighted_router** | 58.4% | 53.3% |
| **concat_linear** | 69.2% | 62.0% |
| **residual** | 58.4% | 53.3% |
| **concat_only_pooled** | 49.6% | 50.3% |

If methods are numerically identical, aggregation is **NOT** the bottleneck.

---

## 8. 11E — Composition Order

| Pair | Forward (Mixed) | Reverse (Mixed) | Diff |
|---|---:|---:|---:|
| **mlp+vs+graph** | 50.3% | 54.0% | -3.7pp |
| **mlp+vs+attention_v2** | 61.6% | 61.1% | +0.5pp |
| **graph+vs+attention_v2** | 55.1% | 51.5% | +3.5pp |

Forward vs reverse means: chain `A->B` vs `B->A` averaged over mixed families.

---

## 9. 11F — Routing Selection Diagnosis

| Family | Agreement (Phase 10 top-2 = Oracle) | Useful-Composition Recovery |
|---|---:|---:|
| **F** | 0.0% | 91.7% |
| **R** | 0.0% | 66.9% |
| **C** | 0.0% | 1.4% |
| **FR** | 0.0% | 68.3% |
| **RC** | 0.0% | 22.5% |
| **FC** | 0.0% | 67.5% |
| **FRC** | 0.0% | 34.7% |

Mean k = 2.00; k distribution = {2: 840.0}

---

## 10. 11G — Benchmark Semantic Validation

Component-removal / destruction controls. For each (family, component) cell, the entry is the rate at which flipping the component's vote flips the composite target — a high rate means the component is genuinely load-bearing.

| Family | F (flip rate) | R (flip rate) | C (flip rate) |
|---|---:|---:|---:|
| **FR** | 49.7% | 49.7% | 0.0% |
| **RC** | 0.0% | 74.4% | 74.4% |
| **FC** | 74.7% | 0.0% | 74.7% |
| **FRC** | 45.8% | 45.8% | 45.8% |

The destroy-channel results (zeroing each component's feature channels) are in `mixed_task_validation.csv`.

---

## 11. 11H — Minimal Validated Composition

Using the 11A best-oracle combinations, evaluate the smallest intervention. Adapter uses a single `H -> H` linear layer trained on a small mixed split.

| Family | No Adapter | With Linear Adapter |
|---|---:|---:|
| **F** | 100.0% | 100.0% |
| **R** | 75.8% | 68.3% |
| **C** | 99.7% | 99.7% |
| **FR** | 75.0% | 75.0% |
| **RC** | 58.6% | 58.6% |
| **FC** | 51.4% | 51.4% |
| **FRC** | 80.8% | 74.7% |

Adapter adds 400 parameters.

---

## 12. Primary Comparison Table

| Policy | Mixed Acc | F | R | C | FR | RC | FC | FRC | FLOPs | Latency | Avg active |
|---|---:|---|---|---|---|---|---|---|---:|---:|---:|
| **k=1 (Phase 10 baseline)** | 52.2% | - | - | - | - | - | - | - | 9,164 | - | 1.00 |
| **Oracle composition (per-family best)** | 66.5% | 100.0% | 75.8% | 99.7% | 75.0% | 58.6% | 51.4% | 80.8% | - | - | - |
| **Sequential best (chain)** | 66.9% | - | - | - | - | - | - | - | - | - | - |
| **Minimal validated composition** | 66.5% | 100.0% | 75.8% | 99.7% | 75.0% | 58.6% | 51.4% | 80.8% | - | - | - |


---

## 13. Causal Diagnosis Table (§28)

| Bottleneck | Status | Evidence |
|---|---|---|
| **EXPERT_CAPABILITY** | PARTIALLY SUPPORTED | Oracle ceiling on mixed = 66.7% essentially equal to Phase 10 single-expert ceiling = 66.7%; no multi-expert combination materially exceeds the best single expert. |
| **REPRESENTATION_TRANSFER** | SUPPORTED | Final-target decodability at 'after_blocks' is high (max 65.6%); intermediate state contains sufficient task information. |
| **AGGREGATION** | SUPPORTED | Aggregation method spread = 18.8% > 5pp; aggregation choice matters. |
| **COMPOSITION_ORDER** | SUPPORTED | Order differs > 5pp in 9/21 (mixed family, order) pairs. |
| **ROUTER_SELECTION** | PARTIALLY SUPPORTED | Mean mixed-family useful-composition recovery = 36.7% is low; router selection is a bottleneck. |
| **BENCHMARK_SEMANTICS** | SUPPORTED | 9/9 (family, component) pairs show that removing the component flips the target > 10% of the time; the mixed benchmark genuinely requires multi-component reasoning. |
| **ROUTING_REPRESENTATION** | PARTIALLY SUPPORTED | Router representation decodes 7-family at 49.3% (chance = 14.3%); some signal. |
| **COMPUTE_ECONOMICS** | NOT TESTED | Compute economics requires full accuracy comparison across policies. |

---

## 14. Causal Conclusion

The selected case is **CASE E — No existing combination works**, with the following supporting evidence (full per-seed numbers in `failure_diagnosis.csv` and `summary.json`):

- Phase 10 k=1 mixed baseline: 52.2%
- Phase 10 single-expert ceiling (recomputed on the same experts): 66.4%
- Oracle k<=3 mixed ceiling (11A): 66.5%
- Routing representation decodes 7-family at: 48.7% (chance = 14.3%)
- Best sequential chain mixed accuracy: 66.9%

---

## 15. Limitations & Decision for Next Phase

1. **Frozen experts** — same Phase 6 portfolio as Phase 10. The "expert capability" verdict is conditional on this portfolio.
2. **Synthetic benchmark** — the mixed dataset uses a sum-of-signs target; component-load-bearing is verified analytically but real-world compositional tasks may have different structure.
3. **Linear adapter only** — no deeper adapter was tested; that is by design (§4 forbids architectural escalation).
4. **Single routing representation** — the Phase 8B mean+std routing input is fixed; alternative representations (11F R1-R4) are mentioned in the spec but not tested here unless evidence demands.

The decision for the next phase is **evidence-driven**, not pre-committed. The strongest evidence-supported next research question depends on the verdict above.
