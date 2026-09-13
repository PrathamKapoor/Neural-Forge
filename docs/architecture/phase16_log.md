# Phase 16 Research Log — Embedded Relational Path vs Dedicated Graph Diagnosis

## Starting evidence (from Phase 15)

- Depth-3 improves isolated R reproducibly (H1 SUPPORTED, +4.7pp, 3/3 seeds).
- No compositional transfer (H5 NOT SUPPORTED), no gap closure (H7 NOT SUPPORTED),
  no ceiling increase (H8 NOT SUPPORTED).
- Standalone Graph ≈84–93% R with ≈32–47pp destruction drops vs embedded path ≈5pp.
- Overall Phase 15 verdict: CASE B (propagation sufficient for isolated R only).

## Research question

Why is relational structure strongly exploited by the dedicated Graph specialist but
weakly exploited by the relational path embedded inside JointCo?

## Protocol

- Seeds 11, 23, 37; same benchmark / training / evaluation protocol as Phase 15.
- 16A baselines reuse the Phase 15 weight cache (executed measurements, evals re-run);
  reproduction deltas vs Phase 15 are exactly 0.0pp (deterministic protocol).
- 16F/16G use a custom trainer mirroring `_train_joint_on_mixed` exactly
  (AdamW lr=0.003, wd=1e-4, 40 epochs, best-val checkpoint on the same val split).
- 16C/16D/16E/16I train only small heads on frozen states under one shared head
  protocol (Linear H→2, AdamW 1e-2, 10 epochs; RNG fork/restore deterministic).
- Router frozen. No learned adjacency. No new block code (no intervention earned one).

## 16A — Exact baselines (executed)

| Condition | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| standalone Graph | 50.0% | 89.2% | 54.2% | 49.2% | 52.2% | 48.9% | 52.2% |
| JointCo baseline | 100.0% | 57.8% | 98.9% | 76.9% | 53.9% | 49.4% | 79.4% |
| JointCo depth-3 | 100.0% | 62.5% | 99.2% | 76.7% | 53.6% | 50.0% | 80.0% |
| rel-focused (B) | 100.0% | 60.8% | 52.2% | 72.8% | 46.7% | 46.4% | 50.3% |
| rel-first (C) | 100.0% | 62.8% | 98.9% | 74.7% | 54.2% | 48.9% | 78.9% |

Note: rel-focused neutralises F/C branches, so its C/FC/FRC collapse is mechanical
(C 52.2%); F stays 100% through encoder+residual+head alone. Its R (60.8%) is the
valid comparison against joint depth-3 (62.5%).

## 16B — Input equivalence

Same raw batch feeds both paths (marker positions identical by construction).
Seed-11 statistics: joint pre-relational input mean −0.026/std 0.350;
native Graph input mean +0.004/std 0.389 (std ratio 0.90, mean shift 0.03,
per-channel L2 1.32). Shapes equal. The comparison therefore tests encoder
transformation, not task framing. Scales are comparable — no gross mismatch.

## 16C — Graph-on-Joint-input (critical experiment)

Identical input (depth-3 encoder output), identical head protocol:

- embedded branch + head R: 69.7% (probe 85.0%)
- trained-Graph-computation + head R: 60.6% (probe 79.4%)
- same-input gap: −9.2pp on R (0/3 seeds positive for the Graph computation)

The trained Graph computation is WORSE on joint input than the embedded branch.
Computation difference is NOT the differentiator in the expected direction —
H2 NOT SUPPORTED. The Graph weights are co-adapted to their native input
distribution and do not transfer.

## 16D — Fresh Graph diagnostic on frozen joint input

Fresh GraphBlock+head trained on joint input: R 57.8% (probe 78.3%),
31.4pp below native Graph. Even a newly-trained Graph computation cannot exploit
the joint pre-relational representation — yet H1 is INCONCLUSIVE (not SUPPORTED)
because the native input probe is equally weak (59.4% vs 57.2%): both encoders
emit weakly-informative states, and the native pipeline's advantage is built
downstream by encoder↔block co-adaptation, not present at the input.

## 16E — Stage analysis (probe + usability + dependence)

| Stage | Probe R | Head R | Drop R |
|---|---|---:|---:|
| pre-relational | 57.2% | 52.2% | +0.0pp |
| post-relational | 85.0% | 68.1% | +15.3pp |
| post-fusion | 83.9% | 60.8% | — |
| native input | 59.4% | — | — |
| final (depth-3) | — | 62.5% | — |

The relational path DOES build causal relational information (post probe 85%,
head-usable 68.1%, destruction-sensitive 15.3pp from a 0.0pp pre baseline).
Fusion preserves probe info (83.9%) but head-usability falls (68.1→60.8).
Notably the diagnostic post-relational head (68.1%) exceeds the model's own
final R (62.5%) — a readout-side conversion gap persists inside the depth-3 model
under a different (mean-pool) protocol; reported as calibration, not as a new claim.

## 16F — Branch-freeze co-adaptation

- Rel-focused minus joint R: −1.7pp (1/3 positive) → joint training does NOT
  suppress relational capability. H3 NOT SUPPORTED.
- Rel-first minus joint R: +0.3pp → warmup adds nothing.
- Answer: the relational branch is not intrinsically suppressed by joint
  optimization under this protocol; its limits reproduce in isolation.

## 16G — Gradient diagnosis (VERIFIED)

Per-epoch probe-batch grad-norm mass: head 1.04, encoder 0.74, contextual 0.52,
feature 0.35, relational 0.18, fusion 0.15. Relational share 6.0% — smallest
branch share but above the 5% starvation criterion (criterion NOT failed).
Signal reaches the branch; contextual dominates branch mass. Diagnostic ratios,
not causal attributions.

## 16H/16J/16K — Ablation, causal metric, composition

- Branch ablation on depth-3 recorded per seed in `branch_ablation.csv`.
- Destruction drops on R: graph ≈39pp ≫ rel_first 11.1pp > baseline 5.3pp >
  depth-3 ≈3.6pp. High-R + high-drop (Graph) differs materially from
  mid-R + low-drop (JointCo variants).
- Strongest condition (rel_first): R +5.0pp but RC +0.3pp, FRC −0.6pp →
  isolated capability improved, compositional transfer unresolved (H6 NOT SUPPORTED).
- Ceiling: 66.8% → 67.3% (+0.5pp; H8 NOT SUPPORTED).

## Hypotheses (programmatic)

- H1 INCONCLUSIVE (same-input gap −9.2pp with native lead +28.6pp: ambiguous —
  neither input-suitability nor computation-difference explains it alone).
- H2 NOT SUPPORTED. H3 NOT SUPPORTED.
- H4 INCONCLUSIVE (pre 57.2% vs native 59.4%: both weak, no differential loss).
- H5 PARTIALLY SUPPORTED (post probe 85.0% vs final R 62.5%: downstream conversion loss).
- H6 NOT SUPPORTED. H7 SUPPORTED (+5.8pp sensitivity via rel_first).
- H8 NOT SUPPORTED.

## Verdict (programmatic): CASE E

Relational information exists but remains difficult to convert into task-usable
mixed reasoning. No architectural intervention (localization only).

## Failure diagnosis (explicit criteria)

Failed: computation_gap (−9.2pp — flagged as inverted gap, correctly handled by
H2 NOT SUPPORTED); pre_computation_loss (57.2%); compositional_transfer_failure;
seed_instability (baseline/rel_focused/rel_first R SDs ≥5pp); latency_regression
(1.86× for rel_first — same architecture as depth-3, attributed to µs-scale CPU
measurement variance, not structure). Not failed: input_mismatch, co-adaptation
suppression, post-computation loss, gradient starvation.

## Interpretation (what the evidence jointly says)

1. The embedded computation is BETTER adapted to its input than the Graph op is
   (69.7% vs 60.6%/57.8% on joint input) — stop blaming the embedded computation.
2. The joint input does not support graph-style exploitation even when freshly
   trained (57.8% vs 89.2% native) — yet inputs are equally uninformative
   pre-computation (57% vs 59% probes). The native advantage is encoder↔block
   co-adaptation, which the shared JointCo encoder (serving three branches and
   the final head simultaneously) does not replicate for the relational path.
3. Joint training is not suppressive (H3 NOT SUPPORTED) and signal reaches the
   branch (6% grad mass) — so the issue is not starvation but the shared
   representation's limited relational affordance combined with a downstream
   conversion gap (H5 PARTIAL).
4. Nothing transfers to RC/FRC. Composition is the open problem.

## STOP condition assessment

Phase 16 localizes but does not resolve: no subsystem earned a structural change
(intervention: none). Per §24 this is a meaningful result, not a failure:
the relational pathway has a validated isolated improvement (depth-3), but its
integration into the adaptive heterogeneous expert remains unresolved.

## Decision for Phase 17 (evidence-driven, not pre-committed)

1. Treat mixed composition (RC/FRC) as the open problem; evidence: every
   intervention to date moves R without moving RC/FRC.
2. If a Phase 17 is run, the warranted directions are: (a) encoder↔relational
   co-adaptation (why the shared encoder cannot learn what the Graph encoder
   learns); (b) the post-relational conversion gap (diagnostic head 68.1% vs
   model head 62.5%). Learned adjacency remains unwarranted (topology exonerated).
3. Alternative scientifically valid decision: STOP the relational track and report
   the boundary.
