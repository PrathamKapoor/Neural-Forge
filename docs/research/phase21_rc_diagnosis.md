# Phase 21 — RC Relational Capacity and Decision Conversion Diagnosis

Status: FAIL CLOSED (reproduction gate failed — protocol §18). Not UNRESOLVED as a scientific claim, but as a protocol state: the measurement pipeline must be fixed before any H1-H8 verdict is valid.

## Protocol enforcement (§1, §5, §18)

Non-negotiable: Reproduce Phase 20 (seeds 11/23/37, repaired phase19-repaired-v1, same weights/protocol) within REP_TOL=3pp before any diagnostic claim.

The reproduction gate (`scripts/phase21_rc_diagnosis.py`) compares 16 established Phase 20 measurements (docs/architecture/phase20_log.md) against the Phase 21 measurement pipeline (`results/metrics/phase21_rc_diagnosis/`).

Result: FAIL CLOSED (22 divergences > 3pp; RC measurements MISSING or 0.0%).

Evidence (Phase 20 log, confirmed measurements):
- RC agree: 98.0% (0.980)
- RC disagree: 1.3% (0.013)
- Pure R (best): 67.5% (depth3)
- RC best: 55.0% (attention_v2)
- FRC best: 76.4% (joint_co)
- Joint RC: 53.3% (53.1% joint_co)

Evidence (broken Phase 21 pipeline):
- Most measurements report 0.0% (agreement, disagreement, counterfactuals, gradient shares, R decodability, family-level accuracies).
- Existing `results/reports/phase21_rc_diagnosis.md` incorrectly reports CASE A (Representation formation bottleneck) despite the failed gate.
- Per §18: this verdict must NOT stand when the gate fails.

## Programmatic verdict (per protocol)

CASE: UNRESOLVED / FAIL CLOSED
Reason: Reproduction gate failed (divergence > 3pp; 16/16 measurements diverged; RC core metrics MISSING/0.0%).
Recommendation (§20): Fix reproduction/debug measurement pipeline before any H1-H8 verdict. Minimal architecture change: NONE (no evidence supports any change).

## What this rules out (per §20)

No claim is made about H1-H8 because the measurement pipeline is broken. Specifically:
- We do NOT claim H1 (capacity limitation) is supported or not.
- We do NOT claim H2 (task alignment) is tested.
- We do NOT claim H3 (contextual dominance) has evidence.
- We do NOT claim H4 (combination causality) is established.
- We do NOT claim H5 (head expressivity) is relevant.
- We do NOT claim H6 (locality) has measurements.
- We do NOT claim H7 (order causality) has controlled results.
- We do NOT claim H8 (intervention) is gated or not gated.

All of these require working measurements, which the broken pipeline does not provide.

## What remains unresolved (§20)

The Phase 21 question (does relational capacity explain RC failure?) requires:
1. Working reproduction gate.
2. Working frozen representation probes (R-component, C-component, RC-target at source/query/fused/final locations).
3. Working gradient/branch contribution instrumentation.
4. Working counterfactual swap evaluation (agreement/disagreement split with change rate + correctness rate).
5. Working frozen diagnostic heads (small linear/nonlinear on frozen representations, NOT new architecture).

None of these are available from the broken artifact set. Once the pipeline is repaired (partial rebuild or measurement script fix), the protocol steps (§2-§13) can proceed without any architecture change.

## Phase 22 recommendation (§20, §21)

Do NOT strengthen representation formation (or any architecture) until:
- The reproduction gate passes (all Phase 20 measurements within 3pp on 3/3 seeds).
- H2-H7 measurements are actually executed (not marked NOT TESTED due to pipeline failure).
- H1 provides a directional, 3/3-consistent capacity signal before any depth-4 or larger model is considered.

Only after these gates should any minimal validated capacity increase (if H1 supported + R not starved) be considered.

## Artifacts

- scripts/phase21_rc_diagnosis.py (reproduction gate, FAIL-CLOSED)
- tests/phase21/test_reproduction_gate.py (reproduction verification)
- Existing Phase 20 artifacts preserved (docs/architecture/phase20_log.md, results/reports/phase20_repaired_portfolio.md, results/metrics/phase20_repaired_portfolio/)
- Existing Phase 21 artifacts preserved but marked broken (results/metrics/phase21_rc_diagnosis/, results/reports/phase21_rc_diagnosis.md — broken measurements retained for audit; verdict corrected to FAIL CLOSED)
- No new architecture (no router, no expert, no attention, no NAS, no meta-learning, no RL, no Transformer)
- Phase 5 artifacts untouched (286 existing tests preserved; no benchmark modification)
