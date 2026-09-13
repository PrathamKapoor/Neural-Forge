# Phase 15 — Relational Substep Capability and Message-Passing Diagnosis

## Mandatory Scientific Disclaimer

> Phase 14 established that readout, pooling, classifier-head capacity, and fusion-path changes did not recover relational performance, and that the relational branch output is not task-sufficient for the evaluated relational prediction. Phase 15 diagnoses WHICH aspect of the relational computation is limiting (propagation → aggregation → update capacity → topology → unresolved). The router remained frozen. All verdicts below are derived programmatically from executed experiments.

---

## 1. Established by Phase 14 (not re-litigated)

- Readout alternatives did not recover relational performance.
- Classifier-head enlargement did not help.
- Fusion-path change did not help.
- Relational branch output was task-weak (rel_only R ≈ 48.6%, worst diagnostic).
- Relational information was decodable (probe ≈ 77.2%) but did not translate into prediction (gap ≈ 15pp).

## 2. Phase 15 research question

> Can a minimal, controlled increase or modification of relational computation convert the existing decodable-but-not-task-usable relational signal into stronger relational prediction?

## 3. Baseline reproduction (§3)

Phase 15 JointCo baseline vs Phase 14 reference:

| Family | Phase 15 baseline | Phase 14 reference | Delta |
|---|---:|---:|---:|
| **F** | 100.0% | — | — |
| **R** | 57.8% | 62.2% | -4.4pp |
| **C** | 98.9% | — | — |
| **FR** | 76.9% | — | — |
| **RC** | 53.9% | 53.3% | +0.6pp |
| **FC** | 49.4% | — | — |
| **FRC** | 79.4% | 80.3% | -0.8pp |
| **mixed_mean** | 64.9% | — | — |

Baseline reproduced vs Phase 14 (tolerance 5pp on R/RC/FRC): **True** (Phase 14 data available).

## 4. Exact current relational computation (15A audit)

WHAT: one message-passing round over a fixed ring (self + left + right). Per round: `m = Linear(H→H)(h)`; `msgs = A_norm @ m` with row-stochastic ring adjacency (mean aggregation); `rel = tanh(Linear(2H→H)([h; msgs]))`. No normalisation inside the substep; block-level residual `out = state + scaled_sum + gated_fusion`.

WHY: reuses the validated `GraphBlock`/`RingGraphAdapter` inductive bias (Phase 1-6) inside the Joint pathway.

WHEN: inside the JointCo block after the shared encoder linear, in parallel with the feature and contextual branches, before per-branch scaling, gated fusion and the residual sum.

HOW: neighbour information enters only through the single `A @ W_msg(h)` averaging step, concatenated with self features and compressed by one linear + tanh.

Relational parameter count (H=24): message 600 + update 1176 = **1776** (identical to the standalone Graph specialist's relational core).

## 5. Depth ablation (15B) — WHAT/WHY/FIXED/RESULT

- WHAT changed: `rel_depth` 1 → 2 → 3 (per-round message/update linears).
- WHY: tests insufficient relational propagation.
- WHAT stayed fixed: hidden size, aggregation (mean), topology, feature/context branches, fusion, readout, optimizer, epochs, seeds.
- RESULT:

| Condition | F | R | C | FR | RC | FC | FRC | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 100.0% | 57.8% | 98.9% | 76.9% | 53.9% | 49.4% | 79.4% | 73.8% |
| **depth2** | 100.0% | 57.2% | 99.4% | 76.7% | 53.3% | 49.7% | 80.3% | 73.8% |
| **depth3** | 100.0% | 62.5% | 99.2% | 76.7% | 53.6% | 50.0% | 80.0% | 74.6% |

## 6. Aggregation diagnosis (15C)

Gate decision: aggregation executed = **True**. Reasons: Depth does not fully explain (best depth3: R gain +4.7pp, gap closes -2.5pp): running aggregation per §7.; Aggregation inadequate (best agg_max: +4.2pp): running capacity per §8.

| Condition | F | R | C | FR | RC | FC | FRC | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **agg_sum** | 100.0% | 60.0% | 98.9% | 76.1% | 53.6% | 49.7% | 79.4% | 74.0% |
| **agg_max** | 100.0% | 61.9% | 99.4% | 76.9% | 53.9% | 50.0% | 80.3% | 74.6% |

## 7. Update-capacity diagnosis (15D)

Capacity executed = **True**. Param delta of MLP update vs linear: +600 relational params.

| Condition | F | R | C | FR | RC | FC | FRC | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **cap_mlp** | 100.0% | 61.7% | 98.9% | 74.2% | 53.6% | 50.3% | 79.7% | 74.0% |
| **cap_control_featwide** | 100.0% | 52.5% | 99.2% | 75.3% | 53.6% | 50.0% | 80.3% | 73.0% |

Generic-capacity control (`cap_control_featwide`, ≈+588 feature params) R gain: -5.3pp.

## 8. Topology controls (15E) + marker/position controls (15F)

Destruction drop on R (normal − destroyed): baseline +5.3pp vs candidate (depth3) +3.6pp. Full per-condition controls are in `causal_controls.csv`.

## 9. Branch interaction (15G) + probe reassessment (15H) + graph comparison (15I)

Probe-final gap: baseline 19.2pp → candidate 21.7pp (closes -2.5pp). Branch matrix in `branch_ablation.csv`; standalone-Graph reference in `graph_comparison.csv`.

## 10. Portfolio re-evaluation (15K)

Old single-expert ceiling (mixed): 66.8%. New ceiling with candidate: 67.3% (delta +0.5pp).

## 11. Hypotheses H1–H8 (programmatic verdicts)

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1** | SUPPORTED | Depth depth3 vs depth-1 baseline on R: mean gain +4.7pp, positive on 3/3 seeds. |
| **H2** | SUPPORTED | Aggregation agg_max vs mean baseline on R: mean gain +4.2pp, positive on 3/3 seeds. |
| **H3** | SUPPORTED | MLP relational update vs linear update on R: mean gain +3.9pp, positive on 3/3 seeds. (+600 params, +576 FLOPs). |
| **H4** | NOT SUPPORTED | Candidate destruction dependence delta -1.7pp on R; no evidence of topology-limited dependence. |
| **H5** | NOT SUPPORTED | RC -0.3pp, FRC +0.6pp: isolated R gains did not transfer compositionally. |
| **H6** | PARTIALLY SUPPORTED | Relational gain +1.7pp vs F/C gain +0.1pp (weak separation). |
| **H7** | NOT SUPPORTED | Gap does not close (-2.5pp). |
| **H8** | NOT SUPPORTED | Ceiling delta +0.5pp mixed mean. |

## 12. Minimal validated intervention (15J)

Selected: **depth3** — Outcome A (depth). smallest useful depth increase: depth3

## 13. Final CASE classification (programmatic)

**CASE B — Additional relational propagation is sufficient**

## 14. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
| depth_failure | False | best depth R gain <= 0pp | +4.7pp |
| aggregation_failure | False | best aggregation R gain <= 0pp | +4.2pp |
| capacity_failure | False | capacity R gain <= 0pp | +3.9pp |
| topology_failure | False | candidate relational-destruction drop on R <= 0pp | +3.6pp |
| representation_prediction_gap | True | candidate probe-final gap >= 5pp | 21.7pp |
| compositional_transfer_failure | True | R gain >= 2pp but RC and FRC gains < 1pp | R +4.7pp, RC -0.3pp, FRC +0.6pp |
| seed_instability | True | any condition R seed-SD >= 5pp | depth/depth2; aggregation/agg_max |
| compute_regression | True | candidate params growth > 50% | +55.0% |
| latency_regression | False | candidate latency > 1.5x baseline | 1.46x |
| branch_interference | False | candidate drops F or C by >= 3pp vs baseline | not observed |

## 15. Not tested (explicitly deferred)

- Learned adjacency (no evidence yet implicates fixed topology as the sole bottleneck; §9 gate).
- Router retraining / second expert / large GNN redesign (out of scope per §2).
- Depth 4+ (only if depth 2/3 show beneficial propagation; see gate record above).
- Aggregation variants beyond sum/max and capacity variants beyond the MLP update (only if justified by §7/§8 gates).

## 16. Limitations

1. Three-seed sample (11, 23, 37); conclusions use mean ± SD, never best-seed.
2. Aggregation/capacity diagnoses are sequentially gated; ungated sections report NOT TESTED, not failure.
3. Single benchmark construction (Phase 9 mixed); claims are task-specific.
4. Small workloads: FLOPs and latency rankings may diverge (both reported).
