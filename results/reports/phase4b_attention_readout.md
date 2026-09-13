# Phase 4B — Attention readout/interface diagnosis

## Information flow

The unchanged block maps `[B,S,H]` to `[B,S,H]` with one-head global self-attention, projection, and residual; no positional encoding, mask, or normalization occurs within the block. The encoder is tokenwise. The query index is obtained from the actual channel-4 query marker, never a fixed position. Mean pooling occurs only after the final block. Therefore residuals preserve token-local state, but the original head sees only the sequence mean.

## Readout comparison (10 shared epochs; mean ± sample SD)

| Readout | Valid | Broken association | Value shuffle | Permuted |
|---|---:|---:|---:|---:|
| Mean | 48.9% ± 3.3 | 48.1% | 48.8% | 48.9% |
| Query token | 53.5% ± 6.3 | 45.4% | 48.6% | 53.5% |
| Max | 55.0% ± 1.9 | 46.7% | 49.4% | 55.0% |
| Query + global | 53.9% ± 6.3 | 46.4% | 47.4% | 53.9% |

Permutation produces identical predictions because every readout finds the query from its marker. The query and query-global conditions show association-control decreases, but their valid accuracies are weak and unstable. Max pooling also improves weakly, so this is not specific evidence that query readout exposes a robustly encoded retrieved value.

## Depth

Query readout at depth 1/2/3: 53.5% / 60.0% / 49.9%. Depth 2 improves relative to mean but has high variation (67.1%, 56.7%, 56.3%) and was not independently tuned. It is a signal for further optimization study, not capability proof.

## Probe scope and verdict

Frozen representation/value-decoding probes were not completed within this controlled run, so post-attention information availability is **NOT VERIFIED** rather than inferred. Readout bottleneck is **INCONCLUSIVE**: mean pooling is worst, and association-sensitive changes occur for some alternative readouts, but no condition meets a reproducible success criterion. Existing AttentionBlock contextual capability remains **NOT SUPPORTED**; contextual specialization remains **NOT SUPPORTED**. Routing gate stays closed.
