# Phase 15 Research Log — Relational Substep Capability and Message-Passing Diagnosis

## Starting evidence (from Phase 14)

- Phase 14 JointCo baseline: R ≈ 62.2%, RC ≈ 53.3%, FRC ≈ 80.3%, mixed ≈ 65.1%.
- Relational probe ≈ 77.2%; probe → final gap ≈ 15pp.
- Pooling / head-capacity / fusion-path / direct-branch readouts: all NOT SUPPORTED.
- Relational branch alone ≈ 48.6% on R (worst diagnostic).
- Localisation: upstream of the readout path — the relational computation is the next
  experimentally justified subsystem. NOT a proof that the relational operation is the
  sole bottleneck.

## Research question

Can a minimal, controlled increase or modification of relational computation convert the
existing decodable-but-not-task-usable relational signal into stronger relational prediction?

## 15A — Audit of the existing relational computation (from the implementation)

- Input `[B, S, H]` (H=24), shared encoder state.
- Graph: fixed ring, each node ← {self, left, right}; row-stochastic adjacency
  (mean aggregation over the 3-neighbourhood).
- Rounds: exactly 1. Per round: `m = Linear(H→H)(h)`; `msgs = A_norm @ m`;
  `rel = tanh(Linear(2H→H)([h; msgs]))`.
- No normalisation inside the substep (LayerNorm only in the final classifier head).
- Block-level residual: `out = state + scaled_sum + gated_fusion`.
- Relational params: message 600 + update 1176 = 1776 (same core as standalone GraphBlock).
- Applied inside JointCo after the encoder, parallel to feature/context branches,
  before per-branch scaling, gated fusion and the residual sum.
- Neighbour information enters through a single averaged `A @ W(h)` step.

The new `JointCoRelationalBlock` (additive, Phase 1–14 untouched) parameterises exactly
this substep along depth / aggregation / update-capacity. Its depth-1/mean/linear
configuration was verified bit-exact against `JointCoBlock` relational math
(max abs diff 0.0 on random input).

## Experimental protocol (all conditions)

- Same architecture/dataset/train-protocol/optimizer/budget as Phase 14
  (mixed F+R+C, AdamW lr=0.003, best-val checkpoint, frozen at eval).
- Only the relational substep varies. Router frozen. Seeds 11, 23, 37; mean ± SD.
- Sequential gates (§7/§8) evaluated programmatically from interim aggregates.

## Baseline reproduction (§3)

Phase 15 baseline: F=100.0%, R=57.8%, C=98.9%, FR=76.9%, RC=53.9%, FC=49.4%,
FRC=79.4%, mixed=64.9%. Deltas vs Phase 14: R −4.4pp, RC +0.6pp, FRC −0.9pp —
within the 5pp tolerance (`reproduced: true`). Baseline probe gap 19.2pp.
Note: baseline R itself varies across seeds (50.0/60.0/63.3, SD 6.9pp).

## 15B — Depth ablation (only depth changes)

| Condition | R | RC | FRC | Mixed |
|---|---:|---:|---:|---:|
| baseline (d1) | 57.8% | 53.9% | 79.4% | 64.9% |
| depth2 | 57.2% | — | — | — |
| depth3 | 62.5% | 53.6% | 80.0% | — |

- depth3 R gain +4.7pp, positive on 3/3 seeds (per-seed gains +10.0/+3.3/+0.8).
- depth2 R gain −0.6pp with severe seed instability (43.3/61.7/66.7, SD 12.3pp);
  depth2 is NOT a viable intervention despite sharing the mechanism.
- depth3 seed-SD on R is 2.2pp (stable); F/C preserved (F 100.0→100.0, C 98.9→99.2).
- Gate: depth does not fully explain (R gain +4.7pp but gap widens −2.5pp) →
  aggregation executed per §7.

## 15C — Aggregation (depth fixed at 1)

- agg_sum R=60.0% (+2.2pp); agg_max R=61.9% (+4.2pp, 3/3 seeds).
- Gate: best aggregation +4.2pp < 5pp adequacy bar → capacity executed per §8.

## 15D — Update capacity (depth 1, mean aggregation)

- cap_mlp R=61.7% (+3.9pp, 3/3 seeds; +600 relational params, +576 FLOPs).
- Generic-capacity control (feature widened +588 params): R=52.5% (−5.3pp).
  More parameters do NOT generically help — the relational-path gains are
  specifically relational (supports H6 direction; H6 PARTIALLY SUPPORTED overall
  because F/C deltas are near zero rather than negative).

## 15E/15F — Topology and marker/position controls

- Destruction drop on R: baseline 5.3pp vs depth3 candidate 3.6pp (delta −1.7pp).
  The candidate improves R while being slightly LESS destruction-dependent:
  stronger topology dependence is NOT the mechanism (H4 NOT SUPPORTED).
- Marker neutralisation and token permutation controls recorded per condition in
  `causal_controls.csv`. FRC is marker-sensitive by construction (contextual vote
  requires the marker); the effect applies equally to all conditions.

## 15G — Branch interaction (candidate depth3, seed 11)

F-only R=55.8%, R-only R=53.3%, F+R R=61.7% (≈ full 60.0%): the F+R combination
carries the R improvement; R+C (54.2%) does not. Required statement: isolated
relational capability improved, but compositional transfer was not demonstrated
(RC −0.3pp, FRC +0.6pp; H5 NOT SUPPORTED).

## 15H — Probe reassessment

Baseline probe 76.9% → candidate probe 84.2% (probe rises), final 57.8% → 62.5%;
gap 19.2pp → 21.7pp (widens −2.5pp; H7 NOT SUPPORTED). More relational information
exists under depth3, but the downstream conversion does not keep pace.

## 15I — Standalone Graph comparison (striking reference result)

Standalone Graph specialist: R 84–93% with destruction drops 32–47pp, vs JointCo
baseline R ≈ 58% (drop ≈ 5pp) and depth3 R ≈ 63% (drop ≈ 4pp). The relational
computation embedded inside JointCo is materially weaker and far less
relationally sensitive than the already-existing Graph specialist. Per the spec
this diagnosis must NOT be bypassed by grafting Graph in — it is recorded as the
primary evidence for a Phase 16 direction.

## 15K — Portfolio re-evaluation

Old ceiling (mixed): 66.8% → new ceiling with depth3 candidate: 67.3%
(delta +0.5pp; H8 NOT SUPPORTED). The empirical single-expert ceiling is preserved,
not redefined.

## Compute / latency

- Params: baseline 6464 → depth3 10016 (+55%, flagged by the explicit
  compute-regression criterion — reported, not hidden; the cost is the mechanism).
- Latency: baseline ≈66µs → depth3 ≈96µs per sample (1.46×, under the 1.5× flag).
  FLOPs and latency both reported; rankings agree here.

## Hypotheses (programmatic)

- H1 SUPPORTED (depth3 +4.7pp R, 3/3 seeds).
- H2 SUPPORTED (agg_max +4.2pp R, 3/3 seeds).
- H3 SUPPORTED (mlp update +3.9pp R, 3/3 seeds; generic control −5.3pp).
- H4 NOT SUPPORTED. H5 NOT SUPPORTED. H6 PARTIALLY SUPPORTED. H7 NOT SUPPORTED.
- H8 NOT SUPPORTED.

## Scientific verdict (programmatic): CASE B

**Additional relational propagation is sufficient** — with the mandatory
qualification: sufficient for isolated R (+4.7pp, reproducible), NOT for
compositional RC/FRC transfer (H5 NOT SUPPORTED), NOT for closing the
representation→prediction gap (H7 NOT SUPPORTED), and NOT for raising the
empirical ceiling (H8 NOT SUPPORTED).

Minimal validated intervention: **depth3** (Outcome A — smallest useful depth
increase; depth2 is unstable and helps nothing).

## Failure diagnosis (explicit criteria)

Failures flagged: representation→prediction gap persists (21.7pp);
compositional-transfer failure; seed instability (depth2, agg_max);
compute-regression flag (+55% params). No depth/aggregation/capacity/topology
failure (all improve R), no latency regression, no branch interference.

## Limitations

1. Three seeds; depth3 gains vary in magnitude (+10.0/+3.3/+0.8) though not in sign.
2. depth2's collapse on seed 11 (43.3%) is unexplained — propagation depth interacts
   with optimisation stability; recorded, not hidden.
3. Aggregation/capacity ran under sequential gates at depth 1; interactions
   (e.g. depth3+max) were deliberately not stacked (§2).
4. Task-specific to the Phase 9 mixed benchmark construction.

## Decision for Phase 16 (evidence-driven, not pre-committed)

1. The embedded JointCo relational path (≈63% R, ≈4pp destruction-sensitive) is
   far weaker than the standalone Graph specialist (≈89% R, ≈39pp sensitive) on
   the same data. Phase 16 should diagnose WHY (placement? scale-suppression?
   fusion dilution? single-round bottleneck now partially addressed?).
2. Compositional transfer (RC/FRC) is the unresolved core: no intervention moved it.
3. Learned adjacency remains deferred — topology was exonerated as the mechanism
   (H4 NOT SUPPORTED), so adjacency learning currently has no evidentiary warrant.
4. No router work is warranted by this phase.
