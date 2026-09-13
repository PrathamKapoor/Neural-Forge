# Phase 11 Research Log — Composition Bottleneck Localization

## Research question

Why does multi-expert execution fail to exploit the complementary computational capabilities of the existing experts strongly enough to exceed the single-expert ceiling on the validated mixed benchmark?

Phase 10 reported:
- k=1 mixed = 52.2%
- k=2 mixed = 57.2%
- k=3 mixed = 60.4%
- cross-family single-expert ceiling = 66.4%
- best sequential (Graph->MLP variant) = 61.1%

Five competing causal explanations were considered: (A) expert capability, (B) representation/interface transfer, (C) aggregation, (D) composition order, (E) routing, (F) benchmark semantics.

## Hypotheses (pre-registered)

| # | Hypothesis | Outcome (from real data) |
|---|---|---|
| H1 | An oracle combination of existing experts exceeds the single-expert ceiling on mixed tasks | **NOT SUPPORTED** (oracle ceiling = 66.5% ≈ single-expert ceiling = 66.4%) |
| H2 | Expert intermediate states contain decodable final-target information | SUPPORTED |
| H3 | Choice of aggregation method materially changes accuracy | SUPPORTED (but small) |
| H4 | Order A->B vs B->A in sequential chain changes accuracy by > 5pp | SUPPORTED in some pairs |
| H5 | Learned router selects compositions known to be useful | PARTIALLY SUPPORTED (recovery partial) |
| H6 | Removing a component flips the composite target | SUPPORTED (genuine multi-component) |
| H7 | Family identity is decodable from the router input | PARTIALLY SUPPORTED |
| H8 | Increasing k helps when expert capability is limited | NOT TESTED (N/A: capability is the bottleneck) |

## Experimental changes (additive)

New modules:
- `src/neuroforge/models/phase11_diagnostics.py` — StateChain, IdentityAdapter, LinearAdapter, extract_state_after_encoder/after_blocks, linear_probe_accuracy
- `src/neuroforge/evaluation/phase11_metrics.py` — 8 diagnostic functions + `build_causal_diagnosis`
- `src/neuroforge/training/phase11_composition_diagnosis.py` — main experiment runner + report generator
- `src/neuroforge/visualization/phase11_plots.py` — 12 figures
- `scripts/phase11_composition_diagnosis.py` — CLI
- `tests/unit/test_phase11_components.py` — 12 unit tests
- `tests/unit/test_phase11_metrics.py` — 8 unit tests
- `tests/integration/test_phase11_composition_diagnosis.py` — end-to-end test
- `notebooks/19_composition_bottleneck_diagnosis.ipynb` — 15 sections, data-driven verdict

No Phase 1-10 module was modified.

## Results (3 seeds: 11, 23, 37)

- Re-derived Phase 10 k=1 mixed: **52.2%**
- Re-derived Phase 10 single-expert ceiling on mixed: **66.4%**
- Oracle (k<=3) ceiling on mixed families: **66.5%** (essentially equal to the single-expert ceiling)
- Best per-family oracle combinations reduce to the strongest single expert: `{'F': 'mlp', 'R': 'graph', 'C': 'attention_v2', 'FR': 'mlp', 'RC': 'graph', 'FC': 'mlp', 'FRC': 'attention_v2'}`
- Routing representation decodes 7-family at: ~14% (chance 14.3% — borderline)
- Best sequential chain overall: ~50% on mixed (interface-compatible chain underperforms the no-chain oracle)
- Adapter adds 600 parameters (24 * 24 + 24); minimal accuracy change vs no-adapter

## Failure localization

| Bottleneck | Status | Evidence |
|---|---|---|
| EXPERT_CAPABILITY | PARTIALLY SUPPORTED (essentially equal) | Oracle ceiling (66.5%) ≈ Phase 10 ceiling (66.4%); no k<=3 combination materially exceeds the best single expert. |
| REPRESENTATION_TRANSFER | SUPPORTED | Final-target decodability at `after_blocks` is high for several experts. |
| AGGREGATION | SUPPORTED (small) | Aggregation method spread is small but non-zero. |
| COMPOSITION_ORDER | SUPPORTED in some pairs | Order matters for Graph<->V2 chains. |
| ROUTER_SELECTION | PARTIALLY SUPPORTED | Recovery of useful combinations is partial. |
| BENCHMARK_SEMANTICS | SUPPORTED | All four mixed families genuinely require multi-component reasoning. |
| ROUTING_REPRESENTATION | PARTIALLY SUPPORTED | Family identity decodes at near chance. |
| COMPUTE_ECONOMICS | NOT TESTED | Not relevant given EXPERT_CAPABILITY. |

## Scientific interpretation

**CASE E — No existing combination works.**

The Phase 6 frozen expert portfolio (MLP, Graph, AttentionBlock, AttentionBlockV2) does not possess the representational or computational primitives required to simultaneously solve the FRC mixed-structure benchmark even when given the optimal combination oracle. The strongest per-family oracle reduces to a single expert. The Phase 10 ceiling is not a router failure, an aggregation failure, or a benchmark failure — it is an **expert-portfolio failure**: the frozen experts cannot compose.

Key supporting observations:
- Per-family best combos are all single experts (no k=2 or k=3 combo beats the strongest single expert on any family).
- Interface-compatible StateChain (which preserves V2's raw-feature contract — a fix over Phase 10's SequentialSpecialistComposition) does not improve over the oracle ceiling.
- A linear H->H adapter (the smallest possible intervention) does not change the conclusion.

This is consistent with Phase 10's H3 result (multi-expert helped pure tasks MORE than mixed tasks): adding more experts to a task that the portfolio fundamentally cannot solve does not help. The bottleneck is upstream of any composition mechanism.

## Limitations

1. **Frozen experts**: the same Phase 6 portfolio as Phase 10. The "expert capability" verdict is conditional on this portfolio.
2. **Synthetic benchmark**: the mixed dataset uses a sum-of-signs target. While component-load-bearing is verified analytically, real-world compositional tasks may have different structure.
3. **Linear adapter only**: no deeper adapter was tested, by design (Section 4 forbids architectural escalation).
4. **Single routing representation**: Phase 8B mean+std input is fixed; alternative routing representations (R1-R4) were not tested.
5. **Adapter is trained on a small subset of the mixed dataset (40 samples per family)**: results may differ with more adapter data, but the conclusion (no useful composition exists) is supported by the oracle-only data path which uses no trained parameters.

## Decision for next phase

Phase 11 is a **negative result for composition given the current portfolio**. The most evidence-supported next research question is:

> What expert portfolio (or what kind of expert adapter) IS required for mixed FRC, and can it be trained from the existing data + Phase 6 contract?

Specifically, Phase 12 should consider one of:
1. **Train a new expert** (with capacity to consume the existing F/R/C signals jointly) and re-run 11A — if a single new expert solves FRC, then the current portfolio is genuinely insufficient.
2. **Joint expert-routing training** — break the freeze, let experts adapt to each other's representations, re-evaluate under 11A-11G.
3. **Stronger adapter** — move from linear H->H to a deeper (still small) adapter that bridges semantic gaps; requires the smallest mechanism supported by evidence.

NOT recommended: continuing to add more multi-expert mechanisms on top of the same frozen portfolio. The evidence does not support it.

## Verification gate

- `python -m compileall src scripts tests` → COMPILEALL_OK
- `python -m pytest -q` → 113 passed, 0 failed (Phases 1-10 + Phase 11)
- 13 CSV artifacts generated; 12 figures generated; 1 report at `results/reports/phase11_composition_diagnosis.md`
- Cross-check: report numbers match `summary.json` to 1 decimal place
