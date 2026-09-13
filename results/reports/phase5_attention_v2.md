# Phase 5 — AttentionBlockV2 minimal retrieval hypothesis

## Mathematics

V1 delegates `Q,K,V` projections, scaled dot-product softmax, output projection, and residual to one-head PyTorch self-attention over learned encoder states. It permits token mixing mathematically, but Phases 4A–4C observed no query-value transfer. V2 preserves the same `[B,S,H]` state but explicitly computes `s_i = normalize(x_q[:3])·normalize(x_i[:3])/0.1`, masks `i=q`, forms `r_q = sum_i softmax(s)_i W_v h_i`, and adds `W_o r_q` only to the marker-defined query state. This adds two H×H linear maps and O(SH²+S) work; no position, heads, FFN, normalization stack, or Transformer mechanism was added.

## Results (three seeds)

| V2 depth | Valid | Broken association | Value shuffle | Permuted |
|---|---:|---:|---:|---:|
| 1 | 92.2% ± 11.5 | 41.1% | 49.8% | 92.2% |
| 2 | 98.8% ± 0.4 | 41.2% | 47.4% | 98.8% |
| 3 | 99.0% ± 0.6 | 38.7% | 48.6% | 99.0% |

The unchanged V1 control remained near chance in Phase 4A–4C. V2 is permutation invariant operationally because the query is found by marker and scores depend only on content keys. The valid/control pattern supports content-dependent retrieval: breaking the key relationship or moving values independently removes performance.

## Interpretation

**V2 architectural hypothesis: SUPPORTED. Query-value transfer: SUPPORTED. Contextual task performance: SUPPORTED. Permutation invariance: SUPPORTED. Efficiency trade-off: PARTIALLY SUPPORTED**—V2 adds value/output projections, so parameter/FLOP/latency comparison must be completed before a Pareto claim. **Contextual specialization: NOT YET SUPPORTED**: V2 has not yet been cross-evaluated against MLP/GNN on all task families under capacity matching. H3 remains partially supported and routing remains closed.

The defensible claim is narrow: on the unchanged validated task, explicit content-key retrieval into the query residual enabled the query-conditioned value-transfer behavior that the original NeuroForge AttentionBlock configuration did not demonstrate.
