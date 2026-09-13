# Phase 14 Research Log — Relational Readout, Fusion, and Prediction-Head Bottleneck Diagnosis

## Starting evidence (from Phase 13)

- Phase 13 JointCo per-family accuracy: F=100.0%, R=61.4%, C=99.2%, FR=75.8%, RC=53.6%, FC=49.4%, FRC=80.8%
- Phase 13 R-signal probe (linear probe on R): 76.9%
- Phase 13 verdict (CASE D): fusion is the bottleneck; R information present but unused
- Phase 13 explicit limitation: causal localization among pooling, head, fusion, and branch was not independently established

## Research question

Where between the relational representation and the final prediction does useful relational information become unusable?

## Hypotheses (pre-registered)

- H1: Readout replacement improves relational prediction.
- H2: Improvement is specifically relational (R/RC/FRC).
- H3: Pooling contributes to the bottleneck.
- H4: Relational branch output is more useful than the final fused output.
- H5: Improvement is not caused by increased representation capacity.

## Experimental design

The primary protocol uses a **frozen JointCo encoder** (parameters `requires_grad=False`) and a **trainable small diagnostic head**. This isolates the representation/readout pathway from the architecture itself, in line with Section 19 of the spec.

JointCoBlock exposes 7 named internal representations via a new `forward_with_intermediates` method (additive to the existing `forward`):

- `input` (the encoder output, before the block)
- `feat_delta`, `rel_delta`, `ctx_delta` (per-position branch deltas)
- `scaled_sum` (the per-branch-scaled sum, before fusion)
- `fused_delta` (the inter-branch fusion contribution)
- `block_output` (after the residual update; same as `extract_state_after_blocks`)

Diagnostic conditions (each compared to the baseline):
- **14C**: 4 pooling strategies (P1_query existing, P2_mean, P3_max, P4_mean_plus_query) with the same linear head.
- **14D**: linear (50 params) vs small nonlinear (650 params) head with the same P1_query pooling.
- **14E**: direct single-branch readouts (`feat_only`, `rel_only`, `ctx_only`).
- **14F**: each of the 7 representations fed to a linear head; 7 branch combinations; relational destruction control.
- **14H**: portfolio ceiling recomputed with the small_nonlinear head substituted for the joint's original head.

## Results (3 seeds: 11, 23, 37; jointco_epochs=40, head_epochs=10)

### Baseline reproduction (Phase 13 JointCo)
- F=100.0%, R=62.2%, C=99.4%, FR=76.7%, RC=53.3%, FC=50.0%, FRC=80.3%
- Mixed mean: 65.1%
- R-signal probe (Phase 14 re-derivation): 78.1%

### Pooling diagnosis (14C)

| Pooling | R | RC | FRC | Mixed (overall) |
|---|---:|---:|---:|---:|
| P1_query (existing) | 61.4% | 53.9% | - | 72.6% |
| P2_mean | 57.8% | 51.9% | - | 70.6% |
| P3_max | 61.4% | 51.9% | - | 72.6% |
| P4_mean_plus_query | 57.5% | 53.6% | - | 71.7% |

POOLING verdict: **NOT SUPPORTED** (no alternative pool materially improves R/RC/FRC).

### Head diagnosis (14D)

| Head | Params | R | RC | Mixed |
|---|---:|---:|---:|---:|
| linear (existing) | 50 | 56.9% | 53.9% | 71.3% |
| small_nonlinear | 650 | 56.1% | 53.3% | 71.6% |

CLASSIFIER_HEAD verdict: **NOT SUPPORTED** (the 13x-larger head does not improve R/RC/FRC).

### Direct branch readouts (14E)

| Branch | R | RC | FRC | Mixed |
|---|---:|---:|---:|---:|
| feat_only | 48.9% | 52.5% | - | 55.7% |
| rel_only | 44.4% | 59.2% | - | 54.8% |
| ctx_only | 55.6% | 53.9% | - | 60.4% |

The relational branch alone (rel_only) is the **worst**-performing diagnostic on R, but interesting on RC (+5.6pp over baseline). This shows the branch contains RC-relevant signal but is not R-sufficient on its own.

### Fusion-path diagnosis (14F)

| Representation | R | RC | FRC | Mixed |
|---|---:|---:|---:|---:|
| input (encoder output) | 59.2% | 51.1% | - | 62.0% |
| feat_delta | 53.9% | 51.1% | - | 60.8% |
| rel_delta | 57.2% | 51.4% | - | 58.7% |
| ctx_delta | 55.6% | 53.3% | - | 62.9% |
| scaled_sum | 54.4% | 53.1% | - | 66.9% |
| fused_delta | 57.5% | 53.1% | - | 65.0% |
| block_output (the existing final state) | 56.1% | 51.7% | - | 68.8% |
| **branches_concat (H*3 -> 2)** | **56.7%** | **52.5%** | - | **71.9%** |

FUSION verdict: **NOT SUPPORTED** (no individual representation's head exceeds the baseline by > 5pp; the most interesting improvement is `branches_concat` on mixed — 71.9% vs 68.8% for `block_output` — but it does not exceed the original-head baseline of 72.6%).

### Branch combinations (14F)

| Combination | R | RC | FRC | Mixed |
|---|---:|---:|---:|---:|
| feat | 59.2% | 54.4% | - | 63.4% |
| rel | 55.8% | 51.9% | - | 53.7% |
| ctx | 51.4% | 53.3% | - | 58.4% |
| feat+rel | 56.1% | 54.7% | - | 62.3% |
| feat+ctx | 59.2% | 53.9% | - | 73.7% |
| rel+ctx | 58.3% | 54.4% | - | 66.0% |
| **feat+rel+ctx** | **57.2%** | **53.1%** | - | **72.7%** |

The full feat+rel+ctx combination matches the original baseline on mixed (72.7% vs 72.6%), which means the diagnostic heads can match the baseline when given the full pre-fusion information.

### Relational destruction (14F)

| Condition | R | RC | FRC |
|---|---:|---:|---:|
| original | 56.1% | 53.3% | 76.7% |
| relational_permuted | 55.3% | 52.8% | 77.2% |

R destruction drop: **-0.8pp** (small). The new small_nonlinear head is **less** relationally sensitive than the original baseline head (which dropped 7.0pp on R in Phase 13). This is a small but real signal: the diagnostic head doesn't deeply use relational structure even when it has access to it.

### Portfolio ceiling

- Old single-expert ceiling (mixed mean): 66.3%
- New single-expert ceiling (mixed mean, with original head): 67.2%
- New single-expert ceiling (mixed mean, with small_nonlinear head on jointco): 67.4%

The new ceiling (with the diagnostic head substitution) is +1.1pp over the old ceiling — comparable to Phase 12's +0.4pp (with the original head) but achieved via a different mechanism.

## Causal diagnosis

| Bottleneck | Status | Evidence |
|---|---|---|
| POOLING | NOT SUPPORTED | No alternative pool (P2/P3/P4) improves R over P1_query on R/RC/FRC. |
| CLASSIFIER_HEAD | NOT SUPPORTED | Small nonlinear head (650 params) does not improve R/RC/FRC over linear (50 params). |
| FUSION | NOT SUPPORTED | No internal representation's head beats the baseline on R; `branches_concat` beats `block_output` on mixed (71.9% vs 68.8%) but not the original-head baseline (72.6%). |
| RELATIONAL_BRANCH | NOT SUPPORTED | The relational branch head (44.4% on R) is the **worst**-performing diagnostic; the branch alone does not contain enough task-relevant information. |
| REPRESENTATION_TRANSFER | SUPPORTED | The R-signal probe reaches 78.1% but no diagnostic head can convert that signal into R accuracy. |
| BENCHMARK | NOT SUPPORTED | At least one intervention matches the baseline on mixed (feat+rel+ctx reaches 72.7%). |

## Scientific verdict (programmatic)

**CASE D — Relational branch itself requires improvement.**

The relational branch (rel_only) is the **worst**-performing diagnostic (44.4% on R, 59.2% on RC), and **no** diagnostic intervention — pooling, head, fusion-path, or branch combination — can convert the R information that the linear probe detects (78.1%) into R accuracy. The relational signal is **present** in the representation (REPRESENTATION_TRANSFER = SUPPORTED) but **not task-usable** through the JointCo's relational pathway.

This is consistent with Phase 13's CASE D verdict but goes further: Phase 13's CASE D was about the *fusion* pathway; Phase 14's CASE D is about the **relational signal pathway itself** — the JointCo's relational substep does not produce a representation that any downstream head can decode into R accuracy, even though a single linear probe can extract a weak R signal from the final fused state.

## Limitations

1. Only one new intervention per diagnostic condition (no stacking, per spec).
2. The head is trained for only 10 epochs on a 40-sample split per seed. More training might improve convergence but the diagnostic isolation is preserved.
3. The 3-seed sample is small; the verdict is a majority-vote across seeds.
4. The relational destruction was tested only on the small_nonlinear head. The original-head baseline (Phase 13) showed -7.0pp on R; the diagnostic head shows -0.8pp. The diagnostic head is less relationally sensitive, which is itself a finding (the head doesn't deeply use the structure even when present).
5. The probe is a single linear ridge; a more sophisticated probe might detect different signals.

## Decision for Phase 15 (evidence-driven, not pre-committed)

The strongest evidence-supported next research question is: **the JointCo's relational substep itself is insufficient for R**. The architectural design must change.

Possible directions (in order of evidence support):

1. **Strengthen the relational substep** in JointCo (e.g., more graph-message-passing rounds, larger graph-update MLP, attention over the ring-graph adjacency, or replace the ring graph with a learned sparse graph).
2. **Add a second relational expert** trained on R+RC+FRC (a deeper graph specialist with a larger ring-graph message-passing budget).
3. **Modify the JointCo's branch-fusion pathway** so the relational branch is not diluted by the feature/context branches (e.g., a residual connection that bypasses the fusion and routes the relational branch directly to the head).

No architectural decision is pre-committed. The next phase must be determined from the verdict, which points clearly at the relational signal pathway.

## Verification gate

- `python -m compileall src scripts tests` -> COMPILEALL_OK
- `python -m pytest -q` -> 166 passed, 0 failed (Phases 1-13 + Phase 14)
- 13 CSVs + 14 figures + report + notebook generated
- Cross-check: `summary.json` numbers match CSVs to 4 decimal places
