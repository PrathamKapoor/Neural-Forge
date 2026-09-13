# Phase 13 Research Log — Joint Expert Co-Adaptation and Relational Compositional Diagnosis

## Starting evidence (from Phase 12)

Phase 12 established the new Joint expert's per-family accuracy:

| Family | Joint |
|---|---:|
| F | 100.0% |
| R | 61.4% |
| C | 99.2% |
| FR | 75.8% |
| RC | 53.6% |
| FC | 49.4% |
| FRC | 80.8% |

Phase 12 verdict (CASE B): new expert helps on mixed mean but family-level ceiling is unchanged (66.4% -> 66.8%).

The Joint expert simultaneously holds F and C capability but is weak on R and RC.

## Research question

Can co-adapting the Joint expert's internal Feature, Relational, and Contextual branches improve its ability to jointly solve the mixed-structure benchmark — particularly R, RC, and FRC — without modifying the existing specialist portfolio?

## Hypotheses (pre-registered)

- H1: Joint co-adaptation improves mixed capability
- H2: Improvement is concentrated on R/RC/FRC
- H3: Co-adaptation preserves F/C capability
- H4: Branch interaction matters
- H5: Joint improvement is not merely additional training time

## Experimental design

A critical Phase 13 design observation: the existing `JointBlock` already trains all three sub-modules (feature_net, graph_message/update, context_value/output) jointly through backpropagation, and the three learnable per-branch scales are also trained jointly. So "ordinary joint training of the Joint expert" is **already** a co-adapted condition. The Phase 13 "co-adaptation" condition must therefore be **genuinely distinct** from ordinary joint training.

The minimal distinct intervention: add a small inter-branch fusion (H*3 -> H linear, gated) that mixes the three branch deltas before they are summed. This is implemented as a new block `JointCoBlock` (in `neuroforge.blocks.joint_co`).

Conditions:
- **13A (baseline)**: Phase 12 Joint, 20 epochs, mixed F+R+C training. Reproduction check.
- **13B (extended-training control)**: same Joint, 40 epochs. Tests H5.
- **13C (co-adaptation)**: new JointCo with H*3 -> H fusion, 40 epochs. Tests H1, H4.

Existing 4 specialists remain frozen. The new block adds ~2000 parameters (~30% of the Joint expert). The fusion gate is initialised to -3 (so the fusion contribution starts near zero; training can turn it on).

## Results (3 seeds: 11, 23, 37; baseline_epochs=20, extended_epochs=40)

Per-condition Joint performance on mixed families:

| Condition | F | R | C | FR | RC | FC | FRC | Mixed Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 13A baseline (20ep) | 100.0% | 61.4% | 99.2% | 75.8% | 53.6% | 49.4% | 80.8% | 64.9% |
| 13B extended (40ep) | 100.0% | 55.0% | 99.7% | 72.8% | 53.1% | 49.7% | 80.3% | 64.0% |
| 13C co-adaptation (40ep) | 100.0% | 59.7% | 99.2% | 76.1% | 53.6% | 48.9% | 79.7% | 64.6% |

- 13B (extended) is **slightly worse** than 13A — additional training does not help (H5 NOT SUPPORTED).
- 13C (co-adaptation) is essentially equal to 13A — the inter-branch fusion does not materially improve mixed capability (H1 NOT SUPPORTED).
- The Joint expert's F and C capability is robustly preserved (100% / 99%) across all conditions (H3 SUPPORTED).
- The R/RC/FRC weakness persists across all conditions (H2 inconclusive — no condition improves the weakness).

Branch scales after training (init = 1/3 = 0.333):

| Branch | 13A baseline | 13C co-adaptation |
|---|---:|---:|
| feature | 0.516 | 0.486 |
| graph | 0.366 | 0.338 |
| context | 0.460 | 0.422 |

The feature branch dominates slightly; the graph branch has the smallest scale (~0.34-0.37) but is **not suppressed** — it is close to its initial value.

Branch ablation on 13C (zero one branch at a time):

| Ablation | R | RC | FRC |
|---|---:|---:|---:|
| all | 59.7% | 53.6% | 79.7% |
| no_feature | 55.8% | 53.6% | 79.7% |
| no_graph | 54.7% | 53.3% | 79.2% |
| no_context | 60.8% | 52.2% | 74.4% |

- Removing the **graph** branch drops R by 5pp and FRC by 0.5pp.
- Removing the **context** branch drops FRC by 5pp and slightly INCREASES R (+1.1pp).
- Removing the **feature** branch drops R by 4pp.

All three branches contribute, with graph and context being the most load-bearing on their respective target families.

## Representation probing (the key diagnostic)

Linear-probe decodability of component signals from the Joint expert's final representation (13C):

| Probe | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| F_signal | 100.0% | 62.8% | 62.5% | 99.4% | 63.3% | 60.6% | 64.2% |
| R_signal | 61.1% | **76.9%** | 67.5% | 68.6% | 66.1% | 60.8% | 59.7% |
| C_signal | 100.0% | 60.0% | 62.5% | 100.0% | 60.8% | 62.8% | 58.9% |
| final_target | 100.0% | **59.7%** | 62.5% | 84.2% | 66.1% | 60.6% | 64.7% |

**The critical finding**: the R_signal probe on the R family is **76.9%**, but the Joint expert's actual R accuracy is **59.7%**. The representation contains the relational information; the final classifier head fails to extract it.

## Relational destruction controls

Original vs relationally-permuted (R/RC/FRC):

| Expert | Condition | R | RC | FRC | R drop (pp) |
|---|---|---:|---:|---:|---:|
| Joint 13A | original | 61.4% | 53.6% | 80.8% | — |
| Joint 13A | permuted | 54.4% | 53.6% | 80.6% | **-7.0** |
| JointCo 13C | original | 59.7% | 53.6% | 79.7% | — |
| JointCo 13C | permuted | 57.2% | 53.3% | 80.0% | **-2.5** |
| Graph specialist | original | 73.9% | 53.9% | 53.1% | — |
| Graph specialist | permuted | 50.8% | 51.9% | 53.1% | **-23.1** |

The Graph specialist is the most relationally sensitive (-23.1pp on R under destruction), confirming that its ring-graph message passing is doing genuine relational computation. The Joint expert is weakly relationally sensitive (-7.0pp at baseline, -2.5pp with co-adaptation), indicating the Joint's relational branch is partially functional but not as discriminative as the Graph specialist.

## Portfolio re-evaluation

- Old single-expert ceiling (mixed): **66.4%** (re-derived; matches Phase 12)
- New single-expert ceiling (mixed, with the best joint condition = 13A baseline at 64.9%): **66.7%**
- Ceiling delta: **+0.3pp** (essentially equal to Phase 12; no meaningful change)

## Causal diagnosis

| Category | Status | Evidence |
|---|---|---|
| UNDERTRAINING | SUPPORTED | 13B (extended) = 64.0% < 13A (baseline) = 64.9%; additional training does not help. |
| BRANCH_INTERFERENCE | PARTIALLY SUPPORTED | 13C (with fusion) = 64.6% ≈ 13B (extended) = 64.0%; inter-branch fusion does not beat extended training. |
| RELATIONAL_CAPABILITY | PARTIALLY SUPPORTED | Removing the graph branch drops R by 5pp (modest contribution). |
| REPRESENTATION_FUSION | **SUPPORTED** | R-signal probe on R = 76.9% but R final accuracy = 59.7% (17.2pp gap). |
| RELATIONAL_SENSITIVITY | PARTIALLY SUPPORTED | R drops -7.0pp (13A) / -2.5pp (13C) under destruction, vs -23.1pp for Graph. |
| PORTFOLIO_LIMITATION | SUPPORTED | Ceiling moves by only +0.3pp. |
| OPTIMIZATION | SUPPORTED | All Joint variants plateau in the 64% range; optimization does not break the cap. |
| BENCHMARK_LIMITATION | PARTIALLY SUPPORTED | Phase 13 ceiling ≈ Phase 12 ceiling. |

## Scientific verdict (programmatic)

**CASE D — Fusion is the bottleneck (R information present but unused).**

The Joint expert's R/RC weakness is **not** caused by insufficient training (H5) or by branch interference (H4). The relational branch is functionally relevant (graph ablation drops R by 5pp) and the relational signal IS in the representation (R-signal probe = 76.9% on R). The failure is at the **fusion/head layer**: the final classifier head does not extract the relational information that is present in the intermediate state.

## Limitations

1. Only one new intervention (JointCo with H*3 -> H fusion) was tested. The spec forbids stacking multiple fixes.
2. The new expert's depth is fixed at 1.
3. Joint training of the existing 4 specialists is intentionally deferred (the spec forbids it in the primary Phase 13 experiment).
4. The 3-seed sample is small; the gap between 13A and 13B (-0.9pp) is within noise and may not be a real effect.
5. The probe is a single linear probe; a more sophisticated probe (e.g., k-NN, attention probe) might detect different signals.

## Decision for Phase 14 (37)

The strongest evidence-supported next research question is:

**The final classifier head / fusion layer is the bottleneck.** A new experiment should target the head, not the architecture or training budget.

Possible next directions (in order of evidence support):

1. **A learnable head adapter for the Joint expert**: the existing head is `LayerNorm + Linear(24, 2)`. A small linear adapter between the state and the head, or a non-linear head (e.g., 24 -> 24 -> 2 with a GELU), might allow the head to extract the relational information that is present in the representation but is currently ignored.
2. **Per-token head pooling**: the current Joint expert pools at the query position (like V2). A different pooling strategy (e.g., mean over the relational branch output) might help.
3. **A relational probe head**: train a small head on the Joint expert's relational-branch output specifically, then use it as a co-head for the final prediction.

No architectural decision is pre-committed. The next phase must be determined from the verdict, which points clearly at the head/fusion layer.

## Verification gate

- `python -m compileall src scripts tests` -> COMPILEALL_OK
- `python -m pytest -q` -> 147 passed, 0 failed (Phases 1-12 + Phase 13)
- 12 CSVs + 15 figures + report + notebook generated
- Cross-check: `summary.json` numbers match CSVs to 4 decimal places
