# NeuroForge Foundation Design

## Scope

This first shippable slice tests adaptive allocation of computation on a controlled, input-level mixed-structure classification problem. It intentionally stops short of external datasets, non-differentiable routing, and domain constraints; those are extensions to add only after the core evidence loop works.

## Alternatives considered

1. **Full mixture-of-experts immediately.** Fast to demonstrate sparse routing, but it would not exercise heterogeneous representations or provide a known oracle path.
2. **One model selected per dataset.** Easier, but it cannot test input-level adaptation.
3. **Selected: shared latent state plus heterogeneous blocks on a synthetic mixed task.** Each sample has feature, relational, or contextual structure. A hidden construction label permits an oracle comparison without becoming a router input.

## Architecture

`MixedStructureDataset` emits sequence tensors and labels. `SharedEncoder` builds a shared latent sequence. MLP, attention, and graph-message blocks consume explicit sequence representations and return a shared state plus a per-sample compute estimate. A learned router emits module probabilities, entropy, and an exit probability. During training it uses a differentiable soft mixture; evaluation can use hard top-1 dispatch. Baselines use the same blocks with a fixed policy.

Adapters are intentionally explicit: a vector adapter pools a sequence for the MLP and broadcasts its output; an identity sequence adapter feeds attention; a ring-graph adapter constructs adjacency for the graph block. Their approximate costs are included in estimates.

## Evidence and failure detection

Every evaluation records accuracy, estimated compute, active modules, utilization, entropy, oracle agreement, routing-label mutual information, and latency. Diagnostics flag single-module collapse, all-module behavior, low entropy/random-like routing, and cost exploitation. FLOP-style estimates and latency are reported separately.

## Failure policy

No accuracy or latency advantage is presumed. If adaptive routing does not beat a fixed model on Pareto comparison, the report must say so. Soft routing may improve optimization while yielding no real latency saving because every branch is evaluated; this is explicitly measured and hard dispatch is separately benchmarked.

## Validation

Unit tests cover data determinism, representation contracts, router normalization, cost accounting, diagnostics, and configuration validation. An integration test performs a forward/backward update and verifies all finite gradients. A reproducible smoke experiment creates machine-readable metrics and a manifest.
