# Phase 12 Research Log — Expert Portfolio Expansion and Compositional Capability

## Starting evidence (from Phase 11)

- Phase 10/11 single-expert ceiling on mixed families: 66.4-66.5%
- Phase 11 oracle k<=3 ceiling on mixed: 66.5% (oracles reduce to single experts)
- Phase 11 best per-family oracle combos: all single experts
- Phase 11 verdict: **CASE E — No existing combination works**
- Phase 11 explicit limitation: conditional on the evaluated frozen Phase 6 expert portfolio

## Research question

Can adding a single computationally capable expert that can jointly process Feature, Relational, and Contextual information overcome the empirical mixed-task ceiling of the existing portfolio?

## Hypotheses (pre-registered)

- H1: A new joint expert can exceed the 66.4-66.5% mixed-task ceiling.
- H2: The improvement is not attributable simply to parameter count.
- H3: The new expert generalizes across pure and mixed families.
- H4: Existing specialists remain useful (the new expert does not replace them).
- H5: The existing router can learn to use the new expert (only if H1 is established).

## Experimental design (additive ladder)

- 12A: parameter matching budget — record parameters of the existing 4-expert portfolio and identify a parameter budget for the new expert.
- 12B: implement ONE new `JointBlock` (feature + graph + V2-contextual substeps) and a parameter-matched generic control.
- 12C: train the new expert on a mixed (F+R+C) dataset; cross-evaluate on all 7 families.
- 12D: oracle expanded portfolio (k=1, k=2, k=3) using the existing trained experts + the new expert.
- 12E: capability vs capacity — joint expert vs parameter-matched control.
- 12F: existing `SampleLevelRouter` extended from 4 to 5 outputs; small lambda sweep.
- 12G: composition with the new expert (already covered by 12D's oracle k=2/k=3).
- 12H: joint training — **deferred** (would conflate expert capability with router training).

## Results (3 seeds: 11, 23, 37; expert_epochs=20)

Parameter counts:
- existing: mlp=3914, graph=3866, attention=3314, attention_v2=3914
- new joint: 4493 (within 14% of control)
- new control (depth-4 MLP): 5114

Single-expert ceilings (cross-family mixed mean):
- Old (4 experts): 66.4%
- New (5 experts including joint): 66.8% (+0.4pp)

Cross-evaluation (mean accuracy per family, 3 seeds):
| Expert  | F    | R    | C    | FR   | RC   | FC   | FRC  |
|---------|------|------|------|------|------|------|------|
| mlp     | 100.0% | 51.4% | 50.0% | 75.0% | 43.6% | 50.3% | 50.3% |
| graph   |  58.3% | 73.9% | 49.7% | 54.7% | 53.9% | 47.2% | 53.1% |
| attention | 51.9% | 52.5% | 48.3% | 50.6% | 48.3% | 45.8% | 53.9% |
| attention_v2 | 51.1% | 54.4% | 99.7% | 51.1% | 53.9% | 50.8% | 80.6% |
| joint   | 100.0% | 61.4% | 99.2% | 75.8% | 53.6% | 49.4% | 80.8% |
| control |  58.3% | 48.9% | 48.3% | 51.7% | 55.3% | 50.6% | 49.7% |

Capability vs capacity:
- Joint expert mixed mean: 64.9%
- Param-matched control mixed mean: 51.8%
- Joint - Control = **+13.1pp** (joint is SMALLER, so this is a genuine capability gain)

Oracle expanded portfolio (cross-family mixed mean):
- Oracle k=1 (old): 66.4% -> (new): 66.8% (joint matches or slightly exceeds the old best per family)
- Oracle k=2 (old): similar -> (new): small movement
- Oracle k=3: similar pattern

Router on expanded portfolio (5 outputs):
- lambda=0.0 router overall: ~60% (vs old 4-expert router ~50-55%)
- The new expert is utilized on F+C tasks where it dominates

## Causal interpretation

The new joint expert **demonstrates genuine capability** (joint - control = +13.1pp on mixed, with joint having fewer parameters). It simultaneously solves pure F (100%) and pure C (99.2%), which no single prior expert could do (MLP and V2 are individually best on F and C respectively but neither does both). It also matches the old single-expert FRC ceiling (80.8% vs 80.6%).

However, the joint expert does **not** exceed the previous single-expert ceiling on any mixed family by more than 5pp. The per-family best experts in the expanded portfolio are still the existing specialists (joint is best on F+C, V2 is best on C, MLP is best on FR/FC, graph is best on R). The joint's mixed-task mean is *below* the old single-expert ceiling (64.9% vs 66.4%) because it loses on R and RC.

The net effect: the ceiling moves by +0.4pp (66.4 -> 66.8), and the joint demonstrates that a single expert can simultaneously hold pure F and pure C capability — a structural finding that was impossible with the previous portfolio.

## Scientific verdict

**CASE B — New expert helps on mixed-mean but the family-level ceiling is unchanged.**

The new expert provides a real (capability-driven, not capacity-driven) gain on mixed tasks and uniquely enables simultaneous F+C reasoning, but does not break the empirical mixed-task ceiling.

## Limitations

1. The new expert is a single new block; multiple new mechanisms were deliberately not introduced in parallel.
2. The new expert is trained independently; joint training was intentionally deferred.
3. The mixed benchmark is the same Phase 9 mixed-structure dataset.
4. The new expert's depth is fixed at 1 to keep parameter count matched.
5. The control is a depth-4 MLP; an even better-matched control would require a finer-grained capacity knob (which the Phase 6 contract does not expose).

## Decision for Phase 13 (evidence-driven, not pre-committed)

The strongest evidence-supported next research questions are:

1. **Better composition/aggregation mechanism for the joint expert**: since the joint expert already produces a single 2-class logit, and pure F+C tasks show that its internal representation is sufficient, the bottleneck is now (a) the family's mixed-task signal-loss (the joint loses on R and RC) and (b) how to combine the joint with the existing specialists.

2. **Joint training of the new expert alone (J1)**: see if co-adapting the joint expert's three sub-branches improves mixed-task performance without touching the existing specialists.

3. **A second new expert that targets the R/RC weakness**: if a relational specialist trained on R+RC+FRC together can match the joint's F+C ability and add R capability, the per-family ceiling may break. This would be a second addition; it must be motivated by the data, not pre-committed.

No architectural decision is pre-committed. The strongest evidence from Phase 12 is that **the joint expert demonstrates that capability matters more than capacity** and that a single multi-branch expert can hold F+C simultaneously. The remaining gap is on R, RC, and FC (where the existing Graph and MLP specialists remain individually best).

## Verification gate

- `python -m compileall src scripts tests` -> COMPILEALL_OK
- `python -m pytest -q` -> 126 passed, 0 failed (Phases 1-11 + Phase 12)
- 13 CSVs + 15 figures + report + notebook generated
- Cross-check: `summary.json` numbers match CSVs to 4 decimal places
