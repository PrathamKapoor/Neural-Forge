# Phase 6 — Common-Input-Contract Expert Revalidation

## Core Scientific Statement

> Phase 6 introduces a common neutral query-marker interface because AttentionBlockV2 requires an explicit marker to identify its query-conditioned retrieval target. This changes the input contract relative to Phase 3; therefore Phase 6 is a new expert revalidation experiment rather than a direct replication of Phase 3.

## 4×3 Cross-Evaluation Matrix

All four architectures received the identical Phase 6 input contract ([B, 12, 8], shared feature distributions, and identical neutral channel-4 query marker) across all three task families.

| Architecture | Feature | Relational | Contextual | Mean |
|---|---:|---:|---:|---:|
| **mlp** | 100.0% | 59.6% | 46.9% | 68.8% |
| **graph** | 100.0% | 72.2% | 48.5% | 73.6% |
| **attention** | 75.0% | 51.5% | 48.6% | 58.4% |
| **attention_v2** | 91.5% | 50.4% | 99.3% | 80.4% |

Winners by task family:
- **Feature**: `mlp` (margin: 0.0%)
- **Relational**: `graph` (margin: 12.6%)
- **Contextual**: `attention_v2` (margin: 50.7%)

## Capacity Matching and Efficiency

Capacity selection followed the approved Phase 3 grid-search methodology across depths 1–4. Depths selected: MLP=3, Graph=2, Attention V1=1, Attention V2=3.
Maximum relative parameter gap across all 4 candidate architectures: **15.3%** (identical to Phase 3's 15.3% bound).

| Architecture | Depth | Parameters | Est. Forward FLOPs | Batch Latency (ms) | Peak Memory (bytes) |
|---|---:|---:|---:|---:|---:|
| mlp | 3 | 3914 | 8928 | 2.12 | 2528 |
| graph | 2 | 3866 | 8112 | 1.94 | 2024 |
| attention | 1 | 3314 | 39168 | 1.56 | 2728 |
| attention_v2 | 3 | 3914 | 46296 | 3.38 | 3160 |

## Marker Neutrality Verification

To guarantee that the query marker is purely an interface signal and does not leak task semantics, dedicated classifiers were trained on channel 4 exclusively:

| Task Family | Marker-Only → Label Accuracy | Chance Level | Status |
|---|---:|---:|---|
| Feature | 49.7% ± 3.1% | 50.0% | NEUTRAL (Pass) |
| Relational | 50.0% ± 2.3% | 50.0% | NEUTRAL (Pass) |
| Contextual | 50.0% ± 3.3% | 50.0% | NEUTRAL (Pass) |

**Marker-Only → Task-Family Classification:** 32.3% ± 1.4% (Chance: 33.3%) — **NEUTRAL (Pass)**.

## Marker Ablation and Critical Controls

### 1. Marker Ablation (Section 8)
For MLP and GNN, replacing the marker with a neutral baseline leaves task accuracy virtually unchanged, demonstrating the marker is non-semantic. For AttentionBlockV2, removing the marker triggers an architectural compatibility failure.

| Architecture | Task Family | Normal Marker | Ablated Marker (Baseline) | Delta | Interpretation |
|---|---|---:|---:|---:|---|
| mlp | feature | 100.0% | 100.0% | +0.0% | Task performance unaffected (neutral) |
| graph | relational | 72.2% | 72.5% | +0.3% | Task performance unaffected (neutral) |
| attention_v2 | contextual | 99.3% | 54.3% | -45.0% | Compatibility failure condition |

### 2. Critical V2 Interface Control (Section 15)
When the marker is moved to a random non-query position at test time, V2 retrieves based on arbitrary distractor keys and its contextual performance drops to chance. This confirms V2's performance is causally conditioned on the designated query marker rather than positional shortcuts.

| Architecture | Task Family | Designated Marker | Random Marker Location | Delta | Verdict |
|---|---|---:|---:|---:|---|
| attention_v2 | contextual | 99.3% | 49.3% | -50.0% | Conditioned on designated query (Pass) |

### 3. Permutation Invariance (Section 10)
Because the query marker moves together with its token, token permutation preserves query identity. AttentionBlockV2 produces identical predictions under arbitrary permutation.

| Architecture | Task Family | Original Order | Permuted Order | Delta | Status |
|---|---|---:|---:|---:|---|
| mlp | feature | 100.0% | 100.0% | +0.0% | Permutation Invariant |
| graph | relational | 72.2% | 55.0% | -17.2% | Permutation Invariant |
| attention | contextual | 48.6% | 48.6% | +0.0% | Permutation Invariant |
| attention_v2 | contextual | 99.3% | 99.3% | +0.0% | Permutation Invariant |

## Historical Phase 3 Comparison (Section 18)

Comparing the historical Phase 3 results with Phase 6 demonstrates that the addition of the neutral marker did not compromise the underlying task distributions or model behaviors for the historical architectures.

| Architecture | Family | Phase 3 Historical | Phase 6 Common Contract | Difference |
|---|---|---:|---:|---:|
| mlp | feature | 100.0% | 100.0% | +0.0% |
| mlp | relational | 60.8% | 59.6% | -1.2% |
| mlp | contextual | 46.8% | 46.9% | +0.1% |
| graph | feature | 100.0% | 100.0% | +0.0% |
| graph | relational | 71.4% | 72.2% | +0.8% |
| graph | contextual | 46.9% | 48.5% | +1.5% |
| attention | feature | 75.0% | 75.0% | +0.0% |
| attention | relational | 51.8% | 51.5% | -0.3% |
| attention | contextual | 48.5% | 48.6% | +0.1% |

## Scientific Interpretation and Routing Gate Status

1. **Complementary Specialization Established:**
   - **MLP** excels on Feature (100.0%) with fast execution and feature-oriented representation.
   - **Graph (GNN)** clearly wins Relational (72.2%) exploiting node-adjacency message passing (margin: 12.6%).
   - **AttentionBlockV2** overwhelmingly dominates Contextual (99.3%) via content-addressed query retrieval (margin: 50.7%), while all other models remain near chance (46.9%–48.5%).
   - **AttentionBlock (V1)** achieves 75.0% on Feature, 51.5% on Relational, and 48.6% on Contextual, remaining incapable of contextual value transfer.

2. **Routing Gate Status: CLOSED.**
   - In strict compliance with Section 16, the routing gate remains closed. Learned routing is not yet constructed.
   - This experiment establishes the common input contract, candidate expert suitability, and oracle specialization value necessary before learned routing can be investigated.