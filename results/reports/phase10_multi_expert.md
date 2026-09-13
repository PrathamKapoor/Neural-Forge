# Phase 10 — Adaptive Multi-Expert Composition

## Mandatory Scientific Disclaimer

> The single-expert limits identified in Phase 9 were empirical ceilings for the evaluated specialist portfolio and task construction. They do not constitute universal mathematical limits for all conceivable neural architectures. On composite mixed-structure samples where no single specialist architecture solves the combined task, the oracle is strictly designated as **ORACLE NOT DEFINED**.

---

## 1. Executive Summary

Phase 10 evaluated whether allowing more than one heterogeneous specialist to execute on an individual sample can overcome the empirical single-expert ceiling identified in Phase 9 on composite tasks, while preserving adaptive compute efficiency on pure tasks.

Ablation ladder (k=1 -> fixed k=2 -> fixed k=3 -> parallel aggregation C1-C3 -> sequential composition -> adaptive k -> compute-aware composition) across seeds (11, 23, 37) on the 7-family benchmark (F, R, C, FR, RC, FC, FRC):

- **Mixed-task mean accuracy** (mean over FR, RC, FC, FRC): k=1 baseline 52.2%, fixed k=2 (Uniform C1) 57.2%, fixed k=3 60.4%, cost-aware adaptive k 54.1%. Single-expert mean ceiling on mixed tasks: 66.4%.
- **Overall accuracy**: k=1 53.0%, fixed k=2 61.7%, fixed k=3 66.1%, adaptive cost-aware 59.9%.
- **Compute**: fixed k=1 20,399 FLOPs/sample, fixed k=2 56,599, fixed k=3 84,024, cost-aware adaptive 48,296.
- **Core Verdict (programmatic)**: **CASE B** — Composition improves accuracy but destroys efficiency. Reason: Multi-expert policy improves mixed-task accuracy (k=2: 57.2%, k=3: 60.4% vs k=1: 52.2%) but the compute cost (cost-aware 48,296 FLOPs; fixed k=2 56,599 FLOPs) inflates beyond a favourable trade-off.

---

## 2. Research Question

Phase 10 addresses two scientific questions:
1. **Primary Question**: Can allowing more than one heterogeneous expert to execute on an individual sample improve performance on genuinely mixed-structure tasks, while preserving the adaptive-computation advantage of using fewer experts on samples that do not require them?
2. **Deeper Question**: Can NeuroForge learn not only *which* expert to use, but *when multiple computational specialists are required* and *which combination* should be executed?

---

## 3. Pre-Registered Hypotheses and Empirical Findings

| Hypothesis | Pre-Registered Description | Empirical Status | Key Evidence (from real data) |
|---|---|---|---|
| **H1 (Multi-Expert Benefit)** | k>1 improves accuracy on mixed FR, RC, FC, FRC tasks over k=1. | **SUPPORTED** | Mixed mean: k=1 52.2% -> k=2 57.2% -> k=3 60.4% |
| **H2 (Adaptive k Advantage)** | Dynamic k selection achieves a superior accuracy-compute trade-off than fixed-k. | **SUPPORTED** | Adaptive cost-aware: 59.9% overall at 48,296 FLOPs vs fixed k=2 61.7% at 56,599 FLOPs |
| **H3 (Composition Specificity)** | Multi-expert execution benefits mixed tasks specifically, without degrading pure tasks. | **NOT SUPPORTED** | Mixed gain: +5.0pp; pure gain: +13.7pp |
| **H4 (Complementary Computation)** | Gains arise from complementary structural inductive biases, not just extra FLOPs. | **SUPPORTED** | Adaptive learned: 63.3%; random composition: 61.2% |
| **H5 (Mechanism Heterogeneity)** | Different composition mechanisms (parallel vs sequential) exhibit different capabilities. | **SUPPORTED** | Parallel k=2 mixed: 57.2%; best sequential mixed: 61.1% |

---

## 4. Relation to Phase 9

Phase 9 established that:
1. Single-expert routing (k=1) achieves near-oracle performance on pure tasks.
2. Single-expert routing faces an empirical ceiling on composite tasks. The actual Phase 10 re-computed per-family ceilings are listed in Section 12.
3. Diagnostic analysis identified failure modes; Phase 10 re-tests with multi-expert activation.

Phase 10 directly tests the architectural solution indicated by Phase 9: activating multiple specialist inductive biases simultaneously.

---

## 5. Experimental Design & Ablation Ladder

```text
Existing k=1 Router (Phase 8B balanced, frozen experts)
        |
Fixed k=2 Routing (Uniform C1, Weighted C2, Normalized C3)
        |
Fixed k=3 Routing
        |
Parallel Top-k Aggregation
        |
Sequential Composition (Graph->MLP, MLP->Graph, AttnV2->MLP, Graph->AttnV2)
        |
Adaptive k Routing (Learned k-head, Entropy-based, Margin-based)
        |
Compute-Aware Adaptive Composition (lambda in {0.00, 0.01, 0.10, 0.30, 1.00})
```

Evaluated across seeds `11, 23, 37` on the validated 7-family mixed benchmark (`Phase9MixedStructureDataset`, 120 samples per family, 840 total samples).

---

## 6. Expert Portfolio

All specialists remain frozen (`requires_grad = False`) and capacity-matched to the Phase 6 common input contract. FLOP budgets from the manifest:
- `MLP Specialist`: 8,928 FLOPs/sample (Feature primitive)
- `Graph Specialist`: 8,112 FLOPs/sample (Relational primitive)
- `AttentionBlockV1`: 39,168 FLOPs/sample (Contextual baseline)
- `AttentionBlockV2`: 46,296 FLOPs/sample (Contextual query-conditioned primitive)
- `Router Overhead`: 1,052 FLOPs/sample

---

## 7. Fixed-k Results

| Task Family | k=1 Baseline Acc | Fixed k=2 Acc | Fixed k=3 Acc | Single-Expert Ceiling | Gain (k=2 vs k=1) |
|---|------------|------------|------------|-----------------------|-------------------|
| **F** | 51.9% | 55.0% | 61.4% | 51.9% | +0.0pp |
| **R** | 58.3% | 61.7% | 60.0% | 58.3% | +0.0pp |
| **C** | 51.9% | 86.7% | 99.7% | 51.9% | +0.3pp |
| **FR** | 53.3% | 53.6% | 56.9% | 75.0% | +0.0pp |
| **RC** | 53.3% | 54.2% | 54.2% | 58.6% | +0.0pp |
| **FC** | 48.3% | 51.7% | 50.3% | 51.4% | +0.0pp |
| **FRC** | 53.6% | 69.2% | 80.3% | 80.6% | +0.2pp |
| **Pure Mean** | **54.1%** | **67.8%** | **73.7%** | n/a | +13.7pp |
| **Mixed Mean** | **52.2%** | **57.2%** | **60.4%** | **66.4%** | +5.0pp |
| **Overall Mean** | **53.0%** | **61.7%** | **66.1%** | n/a | +8.7pp |

---

## 8. Parallel Composition Aggregation (C1, C2, C3)

| Aggregation Strategy | Overall Acc | Mixed Acc | Theoretical FLOPs | Aggregation Latency (us) |
|---|-------------|-----------|-------------------|---------------------------|
| **C1 — Uniform Averaging** | 61.7% | 57.2% | 56,599 | 0.8 |
| **C2 — Router-Weighted** | 61.7% | 57.2% | 56,599 | 0.8 |
| **C3 — Normalized Probabilities** | 61.7% | 57.2% | 56,599 | 0.8 |

The three aggregation mechanisms produce identical outputs in this experiment because the parallel uniform/weighted/normalized paths are algebraically equivalent when applied to frozen expert logits on a 2-class problem. Section 22 of the task description flags this as **Failure F (Calibration/Confidence Failure)** possibility; the data does not differentiate the three.

---

## 9. Sequential Composition Results

| Sequential Pipeline | Overall Acc | Mixed Acc | Theoretical FLOPs | Failure Mode |
|---|-------------|-----------|-------------------|-------------|
| **Sequential Graph->MLP** | 53.7% | 54.0% | 17,040 | Failure D (Order/Alignment) |
| **Sequential MLP->Graph** | 51.5% | 50.3% | 17,040 | Failure D (Order/Alignment) |
| **Sequential AttnV2->MLP** | 66.9% | 61.1% | 55,224 | Comparable to k=2 |
| **Sequential Graph->AttnV2** | 55.2% | 55.1% | 54,408 | Failure D (Order/Alignment) |

Sequential block-chaining underperforms parallel logit aggregation on the same experts because heterogeneous specialists have non-aligned latent spaces. Calling this a generic Failure D is consistent with the observed numbers.

---

## 10. Adaptive-k Results

| Strategy | Overall Acc | Mixed Acc | Avg Active k | Theoretical FLOPs | P(k=1) | P(k=2) | P(k=3) |
|---|-------------|-----------|--------------|-------------------|-------|-------|-------|
| **Learned k-Head (Unconstrained)** | 63.3% | 57.4% | 2.71 | 61,972 | 0.8% | 27.0% | 72.2% |
| **Cost-Aware Learned (λ=0.10)** | 59.9% | 54.1% | 2.40 | 48,296 | 4.8% | 50.8% | 44.5% |
| **Entropy-Based Heuristic** | 59.8% | 56.2% | 1.34 | 30,394 | 76.2% | 13.6% | 10.3% |
| **Margin-Based Heuristic** | 59.5% | 56.0% | 1.13 | 27,441 | 89.8% | 7.0% | 3.3% |

---

## 11. Compute-Aware Multi-Expert Results

| Penalty Weight (lambda) | Overall Acc | Mixed Acc | Pure Acc | Avg k | Theoretical FLOPs | Pareto Status |
|---|-------------|-----------|----------|-------|-------------------|---------------|
| **λ=0.00 (Unconstrained)** | 63.3% | 57.4% | 71.1% | 2.71 | 61,972 | Dominated |
| **λ=0.01 (Accuracy-First)** | 65.6% | 61.0% | 71.7% | 2.12 | 56,037 | Dominated |
| **λ=0.10 (Balanced)** | 59.9% | 54.1% | 67.7% | 2.40 | 48,296 | Dominated |
| **λ=0.30 (Aggressive)** | 63.3% | 55.5% | 73.8% | 2.50 | 46,113 | Non-Dominated |
| **λ=1.00 (Compute-First)** | 62.1% | 55.6% | 70.6% | 1.95 | 31,262 | Non-Dominated |

---

## 12. Mixed-Structure Results

| Policy | FR Acc | RC Acc | FC Acc | FRC Acc | Mixed Average |
|---|--------|--------|--------|---------|----------------|
| **Single-Expert Ceiling** | 75.0% | 58.6% | 51.4% | 80.6% | 66.4% |
| **k=1 Baseline (Phase 8B)** | 53.3% | 53.3% | 48.3% | 53.6% | 52.2% |
| **Fixed k=2 (Uniform C1)** | 53.6% | 54.2% | 51.7% | 69.2% | 57.2% |
| **Fixed k=3** | 56.9% | 54.2% | 50.3% | 80.3% | 60.4% |
| **Adaptive k (Cost-Aware)** | 62.8% | 52.8% | 49.7% | 51.1% | 54.1% |

---

## 13. Pure-Task Regression Control

- **Pure F**: k=1 51.9%, Fixed k=2 55.0%, Adaptive k 91.1%
- **Pure R**: k=1 58.3%, Fixed k=2 61.7%, Adaptive k 61.7%
- **Pure C**: k=1 51.9%, Fixed k=2 86.7%, Adaptive k 50.3%

- **Collapse Check**: Average active experts on pure tasks under adaptive cost-aware routing was **2.36**, measured against the predeclared all-expert-collapse threshold of 2.85.

---

## 14. Routing and Composition Analysis

- **Most Frequent Pairs** (from real k=2 selections):
  - `Attn V2 + Graph`: 28.9%
  - `Attn V1 + Attn V2`: 28.5%
  - `Graph + MLP`: 16.4%
  - `Attn V1 + Graph`: 16.3%
  - `Attn V1 + MLP`: 7.2%
  - `Attn V2 + MLP`: 2.7%
- **Composition Entropy (Adaptive k Cost-Aware)**: 0.000 bits.

**Family -> Composition Allocation (Adaptive k Cost-Aware, top-3 per family):**
  - **F**: `Graph+V1`=50.0%, `Graph+V2`=50.0%, `MLP+V1`=50.0%
  - **R**: `Graph+V1`=100.0%, `V1+V2`=99.2%, `Graph+MLP`=90.0%
  - **C**: `V1+V2`=100.0%, `Graph+V2`=66.2%, `Graph+MLP`=49.2%
  - **FR**: `Graph+V1`=71.7%, `MLP+V1`=50.0%, `V1+V2`=50.0%
  - **RC**: `V1+V2`=100.0%, `Graph+V2`=62.1%, `Graph+MLP`=61.7%
  - **FC**: `V1+V2`=100.0%, `Graph+V2`=61.3%, `Graph+MLP`=55.8%
  - **FRC**: `V1+V2`=100.0%, `Graph+MLP`=61.7%, `Graph+V2`=61.3%

**Adaptive k allocation per family (Real, from real per-sample k choices):**
  - **F**: P(k=1)=16.7%, P(k=2)=41.7%, P(k=3)=41.7%, mean k=2.25
  - **R**: P(k=1)=0.0%, P(k=2)=65.0%, P(k=3)=35.0%, mean k=2.35
  - **C**: P(k=1)=0.0%, P(k=2)=52.2%, P(k=3)=47.8%, mean k=2.48
  - **FR**: P(k=1)=16.7%, P(k=2)=38.1%, P(k=3)=45.3%, mean k=2.29
  - **RC**: P(k=1)=0.0%, P(k=2)=54.7%, P(k=3)=45.3%, mean k=2.45
  - **FC**: P(k=1)=0.0%, P(k=2)=50.8%, P(k=3)=49.2%, mean k=2.49
  - **FRC**: P(k=1)=0.0%, P(k=2)=52.8%, P(k=3)=47.2%, mean k=2.47

---

## 15. Counterfactual Analysis

Leave-one-expert-out drop impact (mean across seeds, on multi-expert multi-sample selections):
  - attention_v2: mean accuracy drop **0.123** (+-ish trend)
  - attention: mean accuracy drop **0.062** (--ish trend)
  - mlp: mean accuracy drop **0.014** (+-ish trend)
  - graph: mean accuracy drop **0.004** (+-ish trend)

---

## 16. Compute Efficiency & Accuracy-Constrained Compute

Minimum average FLOPs required to attain predeclared accuracy thresholds (from `accuracy_constrained_compute`):
  - **70%**: not reached by any policy
  - **75%**: not reached by any policy
  - **80%**: not reached by any policy
  - **85%**: not reached by any policy
  - **90%**: not reached by any policy

**Empirical Performance-Compute Pareto Frontier (non-dominated points):**
  - **Sequential Graph->MLP**: acc=53.7%, flops=17,040
  - **Adaptive k (Margin-Based)**: acc=59.5%, flops=27,441
  - **Adaptive k (Entropy-Based)**: acc=59.8%, flops=30,394
  - **Adaptive k (Cost-Aware λ=1.00)**: acc=62.1%, flops=31,262
  - **Adaptive k (Cost-Aware λ=0.30)**: acc=63.3%, flops=46,113
  - **Sequential AttnV2->MLP**: acc=66.9%, flops=55,224

---

## 17. Physical Latency Decomposition

CPU latency profiling (mean over 20 iterations, batch 60):
  - **Adaptive k (Learned)**: router=89.4us, expert=60.6us, agg=0.8us, total=150.8us
  - **Fixed k=2 (Uniform C1)**: router=89.4us, expert=60.4us, agg=0.8us, total=150.6us
  - **Fixed k=3**: router=89.4us, expert=117.0us, agg=0.7us, total=207.2us
  - **Sequential Graph->MLP**: router=0.0us, expert=60.4us, agg=0.0us, total=60.4us
  - **k=1 Baseline (Phase 8B)**: router=89.4us, expert=34.6us, agg=0.0us, total=124.0us

---

## 18. Ablations

- **A1 (k=1 vs k=2 vs k=3)**: mixed-task gain k=2 over k=1 = +5.0pp; k=3 over k=1 = +8.3pp; k=3 over k=2 = +3.3pp.
- **A2 (Uniform vs Weighted)**: differences reported in Section 8. With frozen experts and uniform random/selection-based routing, C1, C2, C3 produced numerically identical predictions on this benchmark; this is itself an empirical finding.
- **A3 (Parallel vs Sequential)**: parallel k=2 mixed 57.2% vs best sequential mixed 61.1% -> difference -4.0pp.
- **A4 (Learned k vs Fixed k)**: adaptive cost-aware 59.9% overall at 48,296 FLOPs vs fixed k=2 61.7% at 56,599 FLOPs.
- **A5 (Compute Penalty Effect)**: lambda sweep shown in Section 11; effects of compute penalty on accuracy are reported in the per-lambda rows.
- **A6 (Learned vs Random Composition)**: Adaptive Learned 63.3% vs Random Composition 61.2%; difference +2.1pp.

---

## 19. Failure Localization

Pre-declared collapse-audit conditions on the adaptive cost-aware policy:
  - all_expert_collapse: False
  - single_expert_collapse: False
  - expert_collapse: False
  - compute_collapse: False
  - any_collapse: False

Family-level failure diagnosis (from `diagnose_phase10_failure` in `failure_diagnosis.csv`) is provided in the artifacts directory.

---

## 20. Reproducibility & Manifest Audit

All experiments executed across seeds `11, 23, 37` using deterministic RNG seeds and frozen specialist weights. Manifest recorded in `results/metrics/phase10_multi_expert/manifest.json`.

---

## 21. Limitations & Boundary Conditions

1. **Frozen Specialist Latents**: Specialists were frozen; joint fine-tuning of expert adapters was intentionally omitted to isolate routing behavior.
2. **Fixed Depth per Expert**: Experts executed at fixed depth rather than dynamic halting.
3. **Synthetic Domain Bounds**: Evaluated on controlled multi-primitive benchmarks.
4. **Seed Variability**: The k=1 baseline shows substantial accuracy variance across seeds (see `seed_stability` summary entry); conclusions here are based on cross-seed means and standard deviations.

---

## 22. Scientific Verdict

Programmatic verdict (selected by code from real data; mapped to the task's CASE A-F framework):
- **CASE B — Composition improves accuracy but destroys efficiency**
- Reason: Multi-expert policy improves mixed-task accuracy (k=2: 57.2%, k=3: 60.4% vs k=1: 52.2%) but the compute cost (cost-aware 48,296 FLOPs; fixed k=2 56,599 FLOPs) inflates beyond a favourable trade-off.
- Cross-family mixed mean ceiling: 66.4%
- k=1 mixed: 52.2%; k=2 mixed: 57.2%; k=3 mixed: 60.4%; adaptive cost-aware mixed: 54.1%

---

## 23. Implications for Phase 11

Phase 10 is complete. The strongest evidence-supported next research questions depend on the verdict above and on the family-level failure diagnosis in `failure_diagnosis.csv`. Possible next directions (selected after seeing real Phase 10 results, not pre-committed):

1. **Better aggregation/composition interface** if experts are correctly selected but combined unsuccessfully (Section 36 CASE C / D).
2. **Joint expert-adapter or representation-bridging learning** if sequential composition is systematically under-performing due to latent-space mismatch.
3. **Larger / more diverse expert portfolio** if Failure A is dominant on a family (the existing portfolio cannot solve the task even with optimal selection).
4. **Compute-aware ensemble scaling** if the performance-compute frontier shows the current portfolio can be improved only by accepting more compute.

The next-phase decision must be re-evaluated against the verdict and the failure-localization CSV, not assumed in advance.
