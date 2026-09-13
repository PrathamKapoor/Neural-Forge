# Phase 17 Research Log — Heterogeneous Information Composition Diagnosis

## Starting evidence (from Phases 15–16)

- Depth-3 improves isolated R (+4.7pp, 3/3 seeds); no RC/FRC transfer, no gap closure, no ceiling move.
- Relational pathway genuinely creates causal information (post probe 85%, head-usable 68.1%, drop 15.3pp from 0pp pre).
- Embedded computation NOT inferior on equivalent input (69.7% vs 60.6%); suppression/starvation NOT supported.
- Overall: CASE E — information exists but does not convert into mixed reasoning.

## Research question

Where does information become incompatible when heterogeneous branches combine?
Target: the composition interface (not stronger experts).

## Protocol

- Subject: frozen depth-3 JointCo; baselines executed via Phase 15/16 caches (identical protocol, evals re-run).
- One shared diagnostic-head protocol everywhere (Linear, AdamW 1e-2, 10 epochs, RNG fork/restore).
- Order chains: frozen Phase 6 specialists via StateChain/identity (expert-level proxy, documented).
- Seeds 11/23/37, mean ± SD. Router frozen. No learned adjacency.

## 17A — Baselines (executed)

JointCo baseline R=57.8%/RC=53.9%/FRC=79.4%; depth-3 R=62.5%/RC=53.6%/FRC=80.0%;
rel_first R=62.8%; standalone Graph R=89.2% (F/C ≈50–54%: a pure relational specialist).

## 17B/17C — Probe matrix (stages × components)

Branch information REMAINS decodable after fusion: pre→fused probe deltas
F −1.7pp / R +1.5pp / C −0.8pp (negative = fused reads BETTER). No component is
destroyed by fusion. H1 NOT SUPPORTED. This rules out Case 2 (fusion destroys
relational info) and points to Case 1/3 territory: information present, mixed
labels poorly predictable.

Branch component probes: F 65.6%, R 61.4%, C 91.2%.

## 17D/17E/17J/17K — Fusion diagnostic and oracle

Shared-protocol heads on frozen depth-3 states (mean over seeds):

- Oracle F+R+C: R 59.7%, RC 53.3% (+0.3pp over fused head), FRC 76.1% (+6.1pp).
- Fused-state head: R 61.4%, drop 5.6pp; oracle drop 4.7pp.
- R-only vs R+C diagnostic heads on FRC (seed 11): 55.0% → 79.2% (C adds value
  under the diagnostic head); on RC: 46.7% → 50.8% (no coexistence gain).
- Production depth-3 R/RC/FRC: 62.5/53.6/80.0 — the diagnostic oracle does not
  beat production on R or RC, and only on FRC (+6.1pp vs the fused diagnostic head,
  still below production 80.0%).

H5 PARTIALLY SUPPORTED (FRC-only gain), H6 SUPPORTED (diagnostic readout beats the
fusion path on FRC by 6.1pp — narrow, FRC-specific usability gap).

## 17F — Scale domination

Max/min branch RMS 2.40x (rel/feat 1.56x, rel/ctx 2.40x): moderate imbalance,
below the 3x severe bar. H2 PARTIALLY SUPPORTED. Documented, not intervened
(normalization without a usability gate would be architecture without evidence).

## 17G — Interaction gains

Worst pair interaction −2.5pp, zero seeds all-negative → H3 INCONCLUSIVE.
No convergent evidence for destructive interference; per §24 the word
"interference" is therefore NOT used as a finding.

## 17H — Composition order (with readout-confound correction)

Raw R spread 28.1pp LOOKED like order mattering — until grouped by final expert:
graph-final chains (F→R 66.7%, C→R 75.6%) vs mlp/v2-final (47–53%). The spread
tracks WHO READS OUT, not order. Same-final-expert orientations: +8.9/+5.3/+2.2pp,
none 3/3-seed consistent (seed 23 flips the graph-final pair by −17.5pp).
H4 PARTIALLY SUPPORTED at most. Decisively: NO order improves RC/FRC (all ≈47–54%),
so order is not a composition solution under the primary 17M criteria regardless.

A naive rule (raw spread + 2/3 best) would have returned CASE D; the confound
analysis corrected it to PARTIAL. This correction is documented here rather than
hidden: the initial derivation did fire CASE D before the control was applied.

## 17I — Component vs mixed (the smoking gun)

Depth-3 accuracy split by component agreement:

- RC agree ≈ 99–100% vs RC disagree ≈ 0–2% (all seeds).
- FRC agree ≈ 97–100% vs disagree ≈ 70–77% (majority vote rescues 3-way cases).

The model knows the components (probes agree) but predicts essentially ONE of
them on RC: "knows R" without "knows R and can combine it with C". This is the
sharpest behavioral localization of the composition failure in the program.

## 17K — Frozen vs production

Production (62.5/53.6/80.0) ≥ diagnostic heads on R/RC; diagnostic oracle only
leads on FRC vs the fused diagnostic head. The limitation is deeper than the
production readout: even the diagnostic head cannot exploit branch information
for RC. (H6's SUPPORTED status is FRC-specific and does not contradict this.)

## 17M/17N — Criteria and ceiling

RC/FRC: no condition improves RC (oracle +0.3pp); FRC +6.1pp diagnostic-only.
Ceiling 66.3% → 67.0% (+0.7pp; H8 NOT TESTED — no candidate).

## Hypotheses (programmatic)

H1 NOT SUPPORTED; H2 PARTIAL; H3 INCONCLUSIVE; H4 PARTIAL; H5 PARTIAL;
H6 SUPPORTED (FRC-specific); H7/H8 NOT TESTED.

## Verdict (programmatic): CASE E

Branch information exists but is not jointly task-usable. NO INTERVENTION —
no single composition mechanism was sufficiently localized (17L gate unmet:
oracle gains did not reach the overwhelming ≥8pp-on-both bar).

## Failure diagnosis (explicit criteria)

Failed: seed_instability (oracle F+R+C RC SD ≥5pp), order_sensitivity flag
(same-final-expert orientation ≥5pp, seed-inconsistent per H4 evidence).
Not failed: fusion loss, scale domination, destructive interaction, insufficient
joint information (oracle DOES add FRC info), compositional transfer (criterion
keyed to R≥2pp which the oracle misses at +1.9pp — boundary case, disclosed).

## Interpretation

1. Fusion does not destroy information; scale imbalance is moderate; interactions
   are weakly negative at worst; order is confounded and RC-irrelevant.
2. The sharpest fact: RC-disagree ≈ 0–2% with RC-agree ≈ 100%. Composition fails
   at the combination step, not the representation step.
3. The diagnostic oracle (+6.1pp FRC-only) shows FRC carries exploitable joint
   signal that RC does not — consistent with majority-vote structure (3-way ties
   impossible) vs RC's tie/ XOR-like combination demand.
4. Nothing earns architecture. The boundary of §25 is met if this replicates:
   heterogeneous capability without task-usable heterogeneous composition.

## Decision for Phase 18 (evidence-driven, not pre-committed)

1. The RC combination step (disagree cases) is the precise open problem —
   any future phase should target RC-disagree accuracy directly.
2. Candidate (unearned) directions, in evidence order: (a) FRC-style joint
   readout mechanisms, since FRC is the only family showing diagnostic joint
   signal; (b) minimal branch normalization (H2 PARTIAL); (c) nothing — report
   the boundary per §25.
3. Learned adjacency, router work, new experts, and fusion redesign remain
   unwarranted by this evidence.
