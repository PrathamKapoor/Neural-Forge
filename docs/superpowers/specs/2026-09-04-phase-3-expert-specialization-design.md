# Phase 3 expert specialization design

## Question and scope

Determine whether the existing `MLPBlock`, `GraphBlock`, and `AttentionBlock`
have reproducible, relative strengths when independently trained as standalone
classifiers. This phase neither modifies block semantics nor retrains a learned
router. Phase 1 and 2 artifacts remain immutable evidence.

## Existing primitive constraints

All experts encode `[B,S,input_dim]` to `[B,S,H]`, apply only repeated copies
of their existing primitive, mean-pool, and use an identical `LayerNorm(H) ->
Linear(H,2)` classifier head. `MLPBlock` discards order through mean pooling;
`GraphBlock` uses the existing fixed ring adapter; `AttentionBlock` is global
self-attention with no positional embedding. There is no new primitive.

## Benchmark

All samples have `[12, 8]` tensors, two balanced classes, and no family ID.
Independent generator seeds make train, validation, and test sets disjoint.

* **Feature:** two pooled channels determine an XOR-like sign interaction.
  The label has no individual-channel correlation and is invariant to token
  permutation and connectivity.
* **Relational:** node feature pairs carry no label signal independently. The
  label is the sign of an aggregate over the values at ring-neighbour pairs;
  a random degree-preserving permutation of adjacency destroys the supplied
  local relation while retaining node features, graph size, and degree.
* **Contextual:** one query key and one matching key token are placed at random
  distinct positions. The target is the sign of the matching token's value.
  Distractor keys/values are balanced. Replacing the query key with a key that
  does not occur preserves token positions and marginal channel statistics but
  destroys the query-to-value association.

The benchmark records construction-grounded expected specialists (MLP, GNN,
Attention), separately from the empirical winner. It includes feature-only
shortcut probes, label/channel correlations, fixed shape checks, and structural
controls. A failed control or successful shortcut is reported, not hidden.

## Capacity matching and fairness

All experts share H=24 and the common encoder/head. A deterministic exhaustive
search over primitive depth 1..4 chooses the triple minimizing maximum relative
parameter gap, then total depth, then lexicographic depth. Only parameter
counts and configuration determine the choice; validation/test scores are never
consulted. The target is a max pairwise gap <=20%; otherwise the closest valid
triple is reported. The common training protocol is AdamW, same initialization
seed policy, batch size, epochs, validation early stopping, and learning-rate
schedule for every cell.

## Measurements and decision gate

For 3 seeds and each architecture x family, serialize per-seed validation/test
accuracy and loss, parameter counts, analytic forward FLOP estimate, activation
elements, warmed CPU batch latency, throughput, peak tracemalloc memory, and
training time. Define within-family rank from mean test accuracy, specialization
margin as best minus second-best mean accuracy, and seed stability as the share
of seeds with the same top-ranked expert.

Oracle selection is allowed only if expected-family winners are stable, structural
controls lower the relevant winner, and no shortcut baseline explains results.
It routes the known family to its independently trained specialist and compares
accuracy/cost against the single strongest fixed expert. It is evidence of
potential selection only, not learned routing.

## Artifacts

Implementation writes only new Phase 3 locations: `configs/phase3.yaml`,
`results/metrics/phase3_expert_specialization/`,
`figures/phase3_expert_specialization/`,
`results/reports/phase3_expert_specialization.md`, and
`notebooks/17_capacity_matched_expert_specialization.ipynb`.
