# Phase 16 — Embedded Relational Path vs Dedicated Graph Diagnosis

## Mandatory Scientific Disclaimer

> Phase 15 established that depth-3 improves isolated relational prediction, but did NOT establish improved mixed relational composition. The large gap between the dedicated Graph specialist and the embedded JointCo relational path is the central diagnostic observation of this phase. The router remained frozen; no learned adjacency was introduced.

---

## 1. Phase 15 established / Phase 15 did NOT establish

- Established: depth-3 improves isolated R reproducibly (H1 SUPPORTED).
- NOT established: improved mixed relational composition (H5 NOT SUPPORTED), gap closure (H7 NOT SUPPORTED), ceiling increase (H8 NOT SUPPORTED).
- Central observation: standalone Graph ≈84–93% R with ≈32–47pp destruction drops vs embedded path ≈5pp drops.

## 2. Research question

> Why does the dedicated Graph expert exploit relational structure strongly while the embedded JointCo relational path does not?

## 3. Exact baselines (16A, executed — not historical numbers)

| Condition | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **standalone Graph** | 50.0% | 89.2% | 54.2% | 49.2% | 52.2% | 48.9% | 52.2% |
| **JointCo baseline** | 100.0% | 57.8% | 98.9% | 76.9% | 53.9% | 49.4% | 79.4% |
| **JointCo depth-3** | 100.0% | 62.5% | 99.2% | 76.7% | 53.6% | 50.0% | 80.0% |
| **rel-focused (B)** | 100.0% | 60.8% | 52.2% | 72.8% | 46.7% | 46.4% | 50.3% |
| **rel-first (C)** | 100.0% | 62.8% | 98.9% | 74.7% | 54.2% | 48.9% | 78.9% |

Reproduction vs Phase 15 (tolerance 5pp): **True**.

## 4. Input equivalence (16B)

Per-seed representation statistics (shape/scale/marker) are in `input_equivalence.csv`.
Key cross-seed comparison (native Graph input vs JointCo pre-relational input):

- shapes_equal: True
- Both paths consume the same raw batch (marker positions identical by construction);
  the comparison therefore tests encoder transformation, not task framing.

## 5. Graph-on-Joint-input (16C, the critical experiment)

Same input representation, different computation, identical head protocol:

- embedded branch + head R: 69.7%
- Graph-on-joint-input + head R: 60.6%
- gap: -9.2pp
- fresh Graph diagnostic on joint input R: 57.8%

## 6. Stage analysis (16E: probe information + task usability + causal dependence)

- pre-relational probe R: 57.2%
- post-relational probe R: 85.0%
- post-fusion probe R: 83.9%
- native-input probe R: 59.4%
- stage head R (pre/post/fusion): 52.2% / 68.1% / 60.8%

## 7. Co-adaptation (16F) and gradients (16G)

- rel-focused minus joint R: -1.7pp
- rel-first minus joint R: +0.3pp
- gradient instrumentation: VERIFIED — rel share 6.0% of grad-norm mass

## 8. Composition (16K) and ceiling

- strongest condition (rel_first): R +5.0pp, RC +0.3pp, FRC -0.6pp vs baseline.
- ceiling: 66.8% → 66.9% (+0.1pp).

## 9. Hypotheses H1–H8 (programmatic)

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1** | INCONCLUSIVE | Same-input gap -9.2pp, native lead +28.6pp: ambiguous. |
| **H2** | NOT SUPPORTED | Same-input computation gap -9.2pp on R; computation is not the differentiator. |
| **H3** | NOT SUPPORTED | Relational-focused training vs joint training on R: -1.7pp. Rel-first: +0.3pp. |
| **H4** | INCONCLUSIVE | Pre-relational probe 57.2%, native 59.4%. |
| **H5** | PARTIALLY SUPPORTED | Post-relational probe 85.0% but final R 62.5%: conversion loss downstream. |
| **H6** | NOT SUPPORTED | RC +0.3pp, FRC -0.6pp: isolated capability did not transfer. |
| **H7** | SUPPORTED | Causal sensitivity increases by +5.8pp on R. |
| **H8** | NOT SUPPORTED | Ceiling delta +0.1pp mixed mean. |

## 10. Final CASE (programmatic)

**CASE E — Relational information exists but remains difficult to convert into task-usable mixed reasoning**

Recommendation: Treat mixed composition (RC/FRC) as the open problem; isolated R gains do not transfer.

Minimal intervention: **none** — No architectural intervention (localization only).

## 11. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
| input_mismatch | False | joint vs native input shapes differ | shapes_equal=True |
| computation_gap | True | graph-on-joint R gain <= 0pp over embedded | -9.2pp |
| coadaptation_suppression | False | rel-focused training >2pp worse than joint (suppression inverted) | -1.7pp |
| pre_computation_loss | True | pre-relational probe < 65% | 57.2% |
| post_computation_loss | False | post-fusion probe >=5pp below post-relational | post 85.0% vs fusion 83.9% |
| compositional_transfer_failure | True | R gain >= 2pp but RC and FRC gains < 1pp | R +5.0pp, RC +0.3pp, FRC -0.6pp |
| seed_instability | True | any R seed-SD >= 5pp | trained/baseline; trained/rel_focused; trained/rel_first |
| gradient_starvation | False | relational grad share < 5% of total | 6.0% |
| latency_regression | True | candidate latency > 1.5x baseline | 1.86x |

## 12. Not tested / deferred

- Learned adjacency (topology exonerated in Phase 15; remains out of scope).
- Router retraining; second expert; Transformer/attention additions.
- Interaction stacks (e.g. depth-3 + max aggregation) — deliberately not stacked.

## 13. Limitations

1. Three seeds; mean ± SD reported, never best-seed.
2. Equivalent-input heads use mean-pool + linear protocol, not the native query/mean heads; native full-model numbers are reported alongside for calibration.
3. Rel-focused training uses the mixed objective with F/C neutralised (single-variable isolation); a pure-relational objective variant was not run.
4. Gradient shares are diagnostic mass ratios on a fixed probe batch, not causal attributions.
