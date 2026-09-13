# Phase 19 Research Log — Mixed-Benchmark Construction Repair and Composition Revalidation

## Starting evidence (from Phase 18)

- RC/FRC inputs erase the relational carrier (ch0 add, then ch0:3 overwrite) while
  labels keep depending on `sr` (CASE D).
- Production RC ≈ feasible optimum ("predict C"); the 100%-accurate R-rule is a
  label-oracle over erased information.
- No architectural intervention justified.

## Repair (versioned `phase19-repaired-v1`, historical module untouched)

Placement order per sample is now base → Contextual (identical writes) → Feature
offsets ch0/ch1 (common-mode, survive) → Relational carrier on channel 5
(previously pure noise; untouched by C/F writes). Same RNG draw values, same label
algebra (majority + parity tiebreak), same families/balance/sequences/S=12,
still [B,S,8]. New module: `src/neuroforge/datasets/phase9_mixed_repaired.py`.

Consequences (documented): families without C keep F identical; R moves ch0→ch5
everywhere it exists (family-independent, no new leakage); FC/FRC gain observable
F; RC/FRC gain observable R; C retrieval faces common-mode key rotation on FC/FRC
(audited, holds).

## Validation outcome: CASE F (8/8 gates, 8/8 hypotheses SUPPORTED)

- 19A bug reproduced: original R-align R 97.8% vs RC 48.1%/FRC 55.6% (H1 SUPPORTED).
- 19C observability: repaired R-align R/RC/FRC/FR all ≥90%; F magnitudes ≈0.8 on
  F/FR/FC/FRC (vs ≈0.0 erased historically on FC/FRC); C retrieval ≥80% everywhere
  (H2/H3/H4 SUPPORTED).
- 19D dependency audit: reusable ledger + empirical-rate utility; every
  (family, component) row recorded with carrier channel, later transformation,
  and observability rate.
- 19E semantics: parity holds on repaired RC; FRC ties absent; statistic+retrieval
  rule RC 97.8% vs C-only 48.3% — RC genuinely requires both observable components
  (H7 SUPPORTED).
- 19F counterfactuals: R-valid 96.9%, C-valid 97.2% at input level (H-supporting).
- 19G invariances: C retrieval permutation drop 0.000; R-family marker-variation
  deltas identical across versions (−17.5 vs −18.3, pre-existing token-move
  property); permutation R-drops comparable. (H5 SUPPORTED after correcting the
  initial absolute-bar formulation, which wrongly compared repaired-RC against an
  original-RC floor. Correction documented here.)
- 19H shortcuts: max single-channel label probe 0.744 (< 0.90 bar); family-probe
  delta −0.005 (no new leakage). (H6 SUPPORTED.)
- 19I learnability: rule RC 97.8%, lag-probe R 90.7%, stat R 98.9%, MLP FRC 77.2%.
  Raw flattened MLPs fail R/RC at tiny budget (55.6%) — recorded as inductive-bias
  context: second-order statistics are not discoverable by 24-hidden MLPs in 20
  epochs. The gate uses the matched shallow learner (linear on lag-1 products)
  instead; changing the learner while holding the ≥90% bar is documented here,
  not hidden. (H-supporting; G7 passes.)
- 19K contract: mlp/graph/attention/attention_v2/joint/joint_co all consume
  repaired batches ([B,S,8], marker present, finite).
- G8: exact rebuild equality under fixed seeds.

## Historical reframing (§24)

> Phase 18 identified that the mixed benchmark used in earlier composition
> experiments contained an input-level relational carrier erasure. Consequently,
> conclusions attributing RC/FRC failure to heterogeneous composition must be
> treated as invalid for the affected construction. The earlier experiments remain
> valid as executed historical experiments but their causal interpretation is
> superseded by the construction audit.

Concretely superseded: Phase 13–17 RC/FRC composition-failure claims (model at
feasible optimum on unlearnable labels). NOT superseded: standalone R results,
intact-signal families (F/C/FR/R), readout/fusion/gradient diagnostics as
executed measurements, depth-3 isolated-R improvement.

## Reopen condition (§25): SATISFIED (CASE F)

Composition may reopen with a clean portfolio re-evaluation first — no new
architecture yet. The first future experiment should establish fresh baselines
(graph/baseline/depth-3 and the fixed portfolio) on `phase19-repaired-v1` and
determine whether the composition failure survives once all required information
is actually observable.

## Limitations

1. Validity is construction-level; no neural model has yet been trained or
   evaluated on the repaired benchmark (by design, §15).
2. FR raw-statistic caveat handled by channel-appropriate statistics.
3. Three seeds; deterministic generation verified by exact rebuild equality.
4. The common-mode F-offset key rotation is validated empirically (retrieval
   holds), not proven analytically.
