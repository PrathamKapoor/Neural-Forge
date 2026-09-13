# Controlled routing diagnosis — Phase 2

## Executive summary

**RQ1 is supported under supervision:** the same router reaches 100% route accuracy on previously unseen samples when given explicit generator metadata. **Natural-input routing is partially supported:** a route classifier using sequence-derived local/end-point statistics reaches 74.8% ± 3.0% (three seeds), well above random (34.0% ± 3.1%) but below oracle. **Emergent task-only routing is not supported as a complete solution:** the straight-through indirect condition reaches 70.4% ± 1.3% route accuracy and 83.5% ± 1.3% predictive accuracy, below the oracle's 100%.

The original Phase 1 negative result remains unchanged. Its route label was derived from row index and was not observable from the router input; that benchmark could not discriminate router failure from identifiability failure.

## Research question and benchmark

Can a learned controller select a generator-defined MLP/graph/attention route for samples in one mixed distribution? Family A has independent positions, Family B an autoregressive local channel-0 motif, and Family C an endpoint-correlated channel-0 motif. The route label is assigned by the generator before models are trained. Labels depend respectively on feature mean, adjacent feature-product, and endpoint difference.

Natural descriptors are `[channel mean, standard deviation, adjacent product, endpoint product]` calculated from each actual sequence. Explicit metadata is the one-hot generator family; it is a diagnostic positive control only, not a deployable input. Three transparent specialist score functions implement those exact three target relations, so the oracle selection has 100% predictive accuracy by construction. This validates the controlled decision problem, but **does not prove that generic trainable MLP/GNN/attention blocks are uniquely optimal**; that separate architecture-suitability study remains required.

## Conditions and results (mean ± sample standard deviation; 3 seeds, 360 unseen test samples/seed)

| Condition | Route accuracy | Prediction accuracy | Proxy compute |
|---|---:|---:|---:|
| Random | 34.0% ± 3.1% | 64.9% ± 2.4% | 2.04 |
| Static MLP | 33.3% ± 0.0% | 66.6% ± 1.4% | 1.00 |
| Oracle | 100.0% ± 0.0% | 100.0% ± 0.0% | 2.00 |
| Natural, supervised hard | 74.8% ± 3.0% | 85.7% ± 2.6% | 2.01 |
| Explicit metadata, supervised hard | 100.0% ± 0.0% | 100.0% ± 0.0% | 2.00 |
| Natural, indirect soft | 70.4% ± 1.3% | 67.0% ± 0.4% | 2.00; all 3 active |
| Natural, indirect hard (ST) | 70.4% ± 1.3% | 83.5% ± 1.3% | 2.00 |
| Small supervised router | 73.5% ± 3.7% | 85.1% ± 2.7% | 2.06 |
| Compute-aware indirect hard | 71.3% ± 0.6% | 84.4% ± 2.2% | 1.98 |

The first-seed natural-supervised confusion matrix shows graph routes perfectly separated (120/120) but MLP and attention confused (78/120 and 76/120 correct). This is expected: the current natural descriptor deliberately measures only structural channel-0 motifs; independent and endpoint motifs overlap more than local correlation.

## Soft, hard, efficiency, and latency

Soft routes use all three specialists, so their active-module count is exactly 3 and their proxy compute is not a physical compute saving. Hard selection uses one specialist in the benchmark instrument. The controller forward pass for a 360-sample CPU batch was approximately 0.52–0.62 ms; because the transparent specialist scores are tiny and are evaluated as a vectorized diagnostic instrument, this is **not a hardware latency claim for the Phase 1 neural blocks**. Phase 1's measured result—hard neural routing slower than graph-only—stands.

## Ablations and failure modes

- **Descriptor information:** explicit metadata removes error completely; natural descriptors do not. The primary bottleneck is information/representation, not inability to optimize the router.
- **Objective:** direct oracle supervision improves route accuracy over indirect task loss (74.8% vs 70.4%). The downstream loss has several acceptable mixtures and does not uniquely identify the oracle route.
- **Architecture:** reducing router hidden width 12→4 modestly reduces accuracy (74.8→73.5%), not a catastrophic capacity failure.
- **Temperature:** 1.0 and 0.5 produced the same hard route decisions in this configuration.
- **Compute penalty:** changes selection/cost only modestly and does not solve MLP/attention confusion.
- **Collapse:** no learned condition collapsed to a single route. Soft indirect routing activated all modules by design; this is an optimization property, not sparse inference.

## Conclusion

| Hypothesis | Status | Evidence |
|---|---|---|
| Router architecture can learn route labels | **SUPPORTED** | Explicit supervised control: 100% on unseen samples. |
| Natural input exposes enough route information | **PARTIALLY SUPPORTED** | 74.8%, with systematic MLP/attention confusion. |
| Task loss alone yields the intended route | **PARTIALLY SUPPORTED** | Above chance, but materially below supervised/oracle routing. |
| Adaptive routing improves Phase 1 efficiency/latency | **NOT SUPPORTED** | This controlled probe does not overturn prior neural latency evidence. |

## Next experiment

Replace transparent specialists with separately trained, capacity-matched MLP, graph, and attention models and first demonstrate family-specific specialization. Improve the natural router representation with relation-aware descriptors available from the typed adapters, then test whether supervised-route gains survive without explicit metadata. Do not pursue dynamic depth until that test is successful.
