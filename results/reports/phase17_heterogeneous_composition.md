# Phase 17 — Heterogeneous Information Composition Diagnosis

## Mandatory Scientific Disclaimer

> Phase 15 established depth-3 improves isolated R. Phase 16 established the relational
> pathway genuinely creates causal relational information, the embedded computation is NOT
> inferior on equivalent input, joint training suppression and gradient starvation are NOT
> supported. Phase 17 asks whether that information can coexist with Feature and Contextual
> information in a jointly usable representation. Router frozen; no learned adjacency.

---

## 1. Scientific chain (Phases 15 → 16 → 17)

- Phase 15: depth-3 improves isolated relational performance (H1 SUPPORTED).
- Phase 16: relational information is genuinely created (probe 85%, drop 15.3pp),
  embedded computation NOT inferior, suppression/starvation NOT supported.
- Phase 17: can heterogeneous information coexist and be jointly exploited?

Information exists ≠ decodable ≠ task-usable ≠ composable. These claims are kept distinct below.

## 2. Baselines (17A, executed)

| Condition | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **JointCo baseline** | 100.0% | 57.8% | 98.9% | 76.9% | 53.9% | 49.4% | 79.4% |
| **JointCo depth-3** | 100.0% | 62.5% | 99.2% | 76.7% | 53.6% | 50.0% | 80.0% |

## 3. Information preservation (17B/17C — probe matrix, stages × components)

| Stage | F_signal | R_signal | C_signal | final_target |
|---|---:|---:|---:|---:|
| **input** | 54.2% | 53.7% | 64.9% | 55.1% |
| **feat_delta** | 65.6% | 55.1% | 65.0% | 65.0% |
| **rel_delta** | 68.8% | 61.4% | 67.0% | 69.1% |
| **ctx_delta** | 51.6% | 54.0% | 91.2% | 65.1% |
| **fused_delta** | 64.4% | 54.9% | 86.5% | 68.6% |
| **block_output** | 67.3% | 59.8% | 92.0% | 76.2% |

Pre→fused probe deltas: F +nanpp,
R +nanpp,
C +nanpp.

## 4. Additive fusion diagnostic (17D) and oracle (17E/17K)

- Oracle (F+R+C preserved) vs fused-state head under one shared protocol:

| Readout | R | RC | FRC | Mixed | Drop R |
|---|---:|---:|---:|---:|---:|
| oracle F+R+C | 59.7% | 53.3% | 76.1% | — | 4.7% |
| fused head | 61.4% | — | — | — | 5.6% |

- Oracle gains over fused head: RC +0.3pp, FRC +6.1pp.

## 5. Scale / domination (17F)

- Max/min branch RMS ratio: 2.40x
- Rel/Feat: 1.56x, Rel/Ctx: 2.40x

## 6. Branch interaction (17G)

Worst pair interaction gain: -2.5pp.
Full pair × family matrix in `branch_interactions.csv`.

## 7. Composition order (17H, frozen StateChains)

- F→R: R 66.7% (drop +12.5pp)
- R→F: R 47.5% (drop +0.0pp)
- R→C: R 51.4% (drop +0.6pp)
- C→R: R 75.6% (drop +26.9pp)
- F→C: R 49.2% (drop +0.0pp)
- C→F: R 52.8% (drop +0.0pp)

Order spread on R: 28.1pp (best: C→R).

Methodological note: the raw spread is confounded by final-expert identity (orders ending
in the Graph specialist read R well regardless of first expert). The controlled test uses
same-final-expert orientations (readout held fixed):

- C-first_graph-final: +8.9pp, positive on 2/3 seeds
- C-first_mlp-final: +5.3pp, positive on 2/3 seeds
- R-first_v2-final: +2.2pp, positive on 2/3 seeds

SUPPORTED additionally requires 3/3-seed unanimity; without it, order is at most PARTIAL.
Critically, no order improves RC/FRC (all ≈47–54%), so order is not a composition solution
under the primary 17M criteria either way.

## 8. Component vs mixed (17I) and joint decodability (17J)

- Agreement-split accuracies and R-only vs R+C vs F+R+C heads are in
  `failure_diagnosis.csv` (splits) and `joint_decodability.csv` (heads).
- Production vs diagnostic heads in `frozen_oracle.csv`.

## 9. Ceiling (17N)

{ceil.get('old_ceiling_mixed_mean', float('nan')) * 100:.1f}% → {ceil.get('new_ceiling_mixed_mean', float('nan')) * 100:.1f}% ({ceil.get('new_minus_old_mixed', float('nan')) * 100:+.1f}pp).

## 10. Hypotheses H1–H8 (programmatic)

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1** | NOT SUPPORTED | Branch information remains decodable after fusion (worst drop -1.7pp). |
| **H2** | PARTIALLY SUPPORTED | Max/min branch RMS ratio 2.40x (moderate imbalance). |
| **H3** | INCONCLUSIVE | Weak negative interaction -2.5pp. |
| **H4** | PARTIALLY SUPPORTED | Same-final-expert orientation C-first_graph-final +8.9pp on R but seed-inconsistent (2/3); raw spread 28.1pp is dominated by final-expert identity. |
| **H5** | PARTIALLY SUPPORTED | Oracle RC 53.3% (+0.3pp), FRC 76.1% (+6.1pp) (modest). |
| **H6** | SUPPORTED | Diagnostic readout exploits jointly available info (RC +0.3pp, FRC +6.1pp) that fusion/readout does not. |
| **H7** | NOT TESTED | No composition mechanism earned an intervention. |
| **H8** | NOT TESTED | No validated candidate. |

## 11. Final CASE (programmatic)

**CASE E — Branch information exists but is not jointly task-usable**

Recommendation: Treat joint usability as the open problem; isolated gains do not compose.

Minimal intervention: **none** — NO INTERVENTION (diagnosis only).

## 12. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
| fusion_information_loss | False | pre→fused probe drop ≥5pp on a component | worst -1.7pp |
| scale_domination | False | max/min branch RMS ≥ 3x | 2.40x |
| destructive_interaction | False | worst pair interaction ≤ −3pp | -2.5pp |
| insufficient_joint_information | False | oracle RC/FRC gains both < 2pp | RC +0.3pp, FRC +6.1pp |
| compositional_transfer_failure | False | R gain ≥ 2pp but RC and FRC gains < 1pp | R +1.9pp, RC -0.6pp, FRC -3.3pp |
| seed_instability | True | any seed-SD ≥ 5pp | oracle/F+R+C |
| order_sensitivity | True | same-final-expert orientation effect ≥ 5pp on R | 8.9pp (seed-consistency in H4 evidence) |

## 13. Not tested / deferred

- Learned adjacency, router work, Transformer/cross-attention/gating/MoE (all out of scope).
- Production fusion redesign (requires an earned gate that was not met, or is reported above).
- Interaction stacks beyond the single-intervention rule.

## 14. Limitations

1. Three seeds; mean ± SD, never best-seed.
2. Diagnostic heads use mean-pool + linear protocol, differing from production query heads; both reported.
3. Order chains compose standalone specialists, not JointCo branches — an expert-level proxy.
4. Component probes are linear (ridge); nonlinear entanglement (Case 4) would need expanded probes.
