# Phase 20 — Clean Portfolio Re-evaluation on Repaired Mixed Benchmark

## Correction first

> The mixed benchmark previously erased the relational carrier from RC/FRC inputs.
> Phase 19 repaired and validated the construction (`phase19-repaired-v1`).
> Phase 20 therefore constitutes the first clean re-evaluation of heterogeneous
> composition under the intended observable task semantics.

### Historical result (invalid construction)

Old mixed-composition experiments showed an RC/FRC composition failure — now
understood as substantially benchmark-induced (model at feasible optimum on
unlearnable labels).

### Repaired result

Below: fresh training on repaired data only; pre-repair checkpoints never used.

### Interpretation

Benchmark-induced changes must NOT be attributed to architecture. Accuracy deltas
vs history reflect task repair unless architecture, protocol, and benchmark are
otherwise controlled (they are not: the benchmark changed).

---

## 1. Cross-evaluation matrix (repaired, mean over seeds)

| Expert | F | R | C | FR | RC | FC | FRC | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mlp** | 99.7% | 52.8% | 49.2% | 76.7% | 45.6% | 88.3% | 57.2% | 66.9% |
| **graph** | 50.0% | 62.8% | 49.2% | 46.9% | 51.9% | 50.0% | 52.2% | 50.3% |
| **attention** | 50.0% | 54.4% | 55.0% | 61.1% | 48.9% | 48.9% | 39.2% | 49.5% |
| **attention_v2** | 41.7% | 46.7% | 99.7% | 63.9% | 55.0% | 55.8% | 63.9% | 59.7% |
| **joint** | 100.0% | 65.6% | 99.7% | 76.1% | 53.3% | 73.1% | 76.1% | 69.7% |
| **joint_co** | 100.0% | 66.1% | 99.7% | 73.9% | 53.1% | 76.1% | 76.4% | 69.9% |
| **depth3** | 100.0% | 67.5% | 100.0% | 71.7% | 52.8% | 72.8% | 75.8% | 68.3% |

## 2. New empirical ceiling (repaired benchmark, evaluated portfolio)

Single-expert mixed: 75.9% → portfolio: 74.6% (-1.3pp).
RC: 57.2% → 55.8%. FRC: 81.1% → 76.1%.
(Historical 66.4% cited for reference only; never used as the new ceiling.)

## 3. Composition, routing, compute-aware sweep

Full per-λ and per-policy tables in `compute_aware.csv`, `fixed_composition.csv`,
`learned_routing.csv`, `random_routing.csv`, `routing_diagnostics.csv`.

## 4. RC/FRC semantics on repaired inputs

Component agreement, input-level R/C swap change-rates, probes, and destruction
in `component_semantics.csv`, `representation_probe.csv`, `causal_controls.csv`.

## 5. Hypotheses H1–H8 (programmatic)

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1** | NOT SUPPORTED | Worst pure 67.5%. |
| **H2** | SUPPORTED | Best-per-family mixed mean 74.1%. |
| **H3** | SUPPORTED | RC margin +46.1pp. |
| **H4** | NOT SUPPORTED | FRC margin -10.8pp. |
| **H5** | NOT SUPPORTED | RC +0.0pp, FRC -0.8pp. |
| **H6** | SUPPORTED | Learned routing beats best fixed expert by +2.5pp mixed. |
| **H7** | SUPPORTED | Single-component collapse persists (agree 98.0% vs disagree 1.3%). |
| **H8** | NOT SUPPORTED | Portfolio vs ceiling -1.3pp mixed (RC -1.4, FRC -5.0). |

## 6. Final CASE (programmatic)

**CASE B — Composition failure persists despite valid observability**

Recommendation: Treat post-repair composition failure as a legitimate research question for Phase 21.

Minimal intervention: **none** — NO INTERVENTION (baseline phase).

## 7. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
| pure_capability_failure | True | a pure family best < 80% | F 100.0%, R 67.5%, C 100.0% |
| mixed_capability_failure | False | best-per-family mixed mean < 60% | 74.1% |
| composition_no_gain | True | fixed composition gains < 3pp on both RC and FRC | RC +0.0pp, FRC -0.8pp |
| routing_no_gain | False | learned routing gain < 2pp over best fixed expert | +2.5pp |
| single_component_collapse | True | agree−disagree gap ≥ 50pp | 96.6pp |
| ceiling_unmoved | True | portfolio vs single-expert ceiling < 2pp | -1.3pp |
| seed_instability | True | any seed-SD ≥ 5pp | RC/oracle_k=2; RC/learned |
| smoke_failed | False | repaired-validity smoke must pass | all seeds pass repaired-validity smoke |

## 8. Limitations

1. Three seeds; mean ± SD, never best-seed.
2. Specialists train on repaired pure-family subsets (120 train); joints on repaired F+R+C.
3. Router trains on all-family repaired mixed; num_experts=7 follows the Phase 12 4→5 precedent (portfolio sizing, not redesign).
4. Task changed vs history: deltas vs Phase 9–18 are repair effects unless otherwise controlled.
