# Phase 18 Research Log — Component Composition and Decision Semantics Diagnosis

## Starting evidence (from Phase 17)

- Branch information survives fusion (probe deltas −1.7/+1.5/−0.8pp).
- RC-agree ≈100% vs RC-disagree ≈0–2% (sharpest behavioral signature).
- Oracle F+R+C helps FRC (+6.1pp) but not RC (+0.3pp).

## Central discovery (18J): input-level erasure of the relational signal

Reading `src/neuroforge/datasets/phase9_datasets.py` (mixed benchmark construction):

- Line 325 adds the relational carrier `cand` to channel 0 when the family contains R.
- Line 336 then **overwrites channels 0:3** with contextual distractors/keys whenever
  the family contains C.

On RC and FRC inputs (R and C both present), the relational carrier is added and
then destroyed. The label still uses `sr` (line 346). Hence `sr` is statistically
independent of the final input tensor on RC/FRC: nothing downstream — no branch,
no head, no architecture — can recover it.

Executed proof (autocorrelation statistic = sign of mean(x0·roll(x0)), which IS sr
when the carrier is present):

| Family | stat-vs-sr agreement |
|---|---:|
| R | 97.8% (statistic validated) |
| RC | 48.1% (erased) |
| FRC | 55.6% (erased) |

(FR reads ~48% for a different, understood reason: the Feature offset dominates the
raw statistic; no overwrite occurs without C, and the model solves FR at 76.7%.)

Consequences, all verified:

1. On RC-disagree, labels equal the R side 59/59 — algebraically, not empirically:
   sc is set deterministically from the sample index (sc=+1 iff i even) and disagree
   labels are the index-parity tiebreak, so y == (sr>0) == NOT(sc>0) identically.
2. The feasible optimum on RC is "predict C" (≈50.8% = agree rate): agree 100%,
   disagree 0%. Production RC 53.6% is AT the feasible optimum, not below it.
3. The R-rule (100% train/test) is a label-oracle: it reads the erased `sr`.
   Judging the model against it is judging against unobservable information.
   The feasible single-component rule is the C-rule, which the model matches
   (train 50.8% vs 50.0%, behavior 99.2%).
4. FRC retains F+C (majority of three with one noise vote): predictable ≈75–80%,
   matching production 80.0% and the diagnostic oracle's FRC-only gain.
5. RC/FRC composition was never testable as constructed. Phases 13–17's
   "composition failure" narrative is corrected, not extended, by this finding.

## Spec-text deviation (documented, not hidden)

The Phase 18 text frames RC as F+R. The construction defines RC from R+C.
All RC component analyses use (sr, sc) per construction; the F component on RC
is a negative control (no F injected → must read ~chance, validating probe honesty).

## Decision behavior under erasure (all convergent)

- Production on RC-disagree matches C 98.8% vs R 1.2%: systematic C-favoring,
  which under erasure is the feasible-optimal policy (H4 SUPPORTED, reinterpreted).
- Counterfactual change-rates: R-swap 7.3% vs C-swap 90.9% (control 7.1%) —
  the diagnostic head demonstrably ignores R and follows C. (The implied-match
  variant scores ~96% for both swaps and is documented as non-discriminative:
  a C-predictor matches implied targets without responding. Change-rate is the
  valid metric; both are reported.)
- Linear AND nonlinear diagnostic heads fail RC-disagree (6.1%/4.5%): correctly
  classified as "deeper than expressivity" — the depth is the erased input.
- R-component best-rep accuracy on RC 48.1% vs C 99.4%; joint (R&C) 43.1%.
- Shuffled controls at ~chance (no accidental correlation).

## Hypotheses (programmatic)

- H1 PARTIAL (C available 99.4%, R 48.1% — missing at input, not in the model).
- H2 NOT SUPPORTED (joint 43.1%).
- H3 SUPPORTED (agree/disagree gap 98.3pp — now read as construction signature).
- H4 SUPPORTED (systematic C-favoring — now read as feasible-optimal).
- H5 NOT SUPPORTED. H6 SUPPORTED (feasible framing: model ≈ C-rule).
- H7 SUPPORTED (erasure demonstrated + parity verified).
- H8 NOT TESTED (18L gate correctly unmet: H2/H5 fail).

## Verdict (programmatic): CASE D

RC/FRC behavior is primarily explained by task semantics (input-level erasure).
NO INTERVENTION — the default and correct outcome: there is no decision mechanism
to fix when the required signal is absent from the input.

## Failure flags (read with the erasure finding)

Failed rows (component/joint unavailable, disagree collapse, favoring, head
inexpressivity, shortcut, counterfactual unresponsiveness) all share ONE upstream
cause: erased R input. They are recorded per the explicit criteria and must NOT
be read as independent model pathologies. `semantic_asymmetry` is a verify row.

## What remains genuinely open (not erased)

1. FRC composition with F+C present (production 80%, oracle diagnostic +6.1pp over
   the fused head): the only family with real joint signal — future composition
   work, if any, belongs here, not on RC.
2. The post-relational conversion gap inside intact-signal families (diagnostic
   head 68.1% vs model head 62.5% on R-family-adjacent states, Phase 16).
3. Whether a corrected benchmark (relational carrier written AFTER contextual
   overwrite, or on disjoint channels) restores RC/FRC testability. Recommended
   as a dataset-construction fix, explicitly NOT as a Phase 18 change
   ("Do NOT redesign the benchmark" was honored).

## Decision for Phase 19 (evidence-driven)

1. Do NOT run another composition-architecture phase against RC: the labels are
   unlearnable-by-construction on disagree and the model is at the feasible optimum.
2. If the program continues: (a) fix the construction order bug (one-line change:
   apply the relational carrier after the contextual overwrite, or reserve
   disjoint channels), re-run 17A–17C as a regression; (b) only then re-ask the
   composition question.
3. Alternative valid decision: STOP the relational-composition track and report
   the boundary — heterogeneous capability demonstrated, composition untestable
   as constructed, feasible-optimal behavior already achieved on RC.
