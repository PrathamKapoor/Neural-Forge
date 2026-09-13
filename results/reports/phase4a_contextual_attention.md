# Phase 4A — Leak-free contextual attention diagnosis

## AttentionBlock audit

`AttentionBlock` consumes and preserves `[B,S,H]`. It uses one-head PyTorch `MultiheadAttention(H, 1, batch_first=True)`, no mask, no positional encoding, no internal normalization, then `Linear(H,H)` and a residual. The standalone wrapper mean-pools only after the block stack and applies `LayerNorm(H) -> Linear(H,2)`.

## Construction and validity

Each sample contains one query-marked token, a different matching key token with the same normalized 3-D content key, normalized distractor keys, and a binary value on the matching key. Three distractor values exactly cancel that matching value globally. The label is the matching token's value. Arbitrary token permutation preserves the oracle label. Maximum-dot-product association oracle is 100% by construction.

## Results (three seeds)

| Model | Valid | Broken association | Value shuffle | Permuted | Length 16 |
|---|---:|---:|---:|---:|---:|
| MLP | 48.6% ± 5.1 | 48.2% | 48.6% | 48.6% | 50.7% |
| GNN | 48.2% ± 4.8 | 47.5% | 48.1% | 48.0% | 49.3% |
| Attention | 49.3% ± 4.0 | 48.5% | 49.9% | 49.3% | 50.4% |

Attention depth 1/2/3: 49.3% / 49.2% / 50.4%. No depth is meaningfully above chance.

## Verdict

Task validity and permutation invariance are **SUPPORTED**. Attention capability, contextual specialization, and long-range generalization are **NOT SUPPORTED** under this budget. Attention-weight diagnostics are **NOT VERIFIED** because the unchanged primitive requests no weights. H3's contextual component remains **NOT SUPPORTED**. Do not proceed to learned routing; diagnose whether mean pooling or optimization prevents use of query-token retrieval before changing the primitive.
