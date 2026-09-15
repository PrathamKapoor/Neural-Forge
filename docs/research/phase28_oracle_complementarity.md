# Phase 28 — Oracle Complementarity & Aggregation Identifiability

Status: `COMPLETE`
Programmatic case: `CASE_D_UNRESOLVED` (evidence approximate; exact requires predictions.npy per architecture)
Architecture: `NONE` | RC: `REMAINS_CLOSED` | Intervention: `NONE` | Gate: `CLOSED`

## Evidence Sources
- `results/metrics/phase25b_reference/` (verified: MLP predictions.npy, checkpoint.pt, manifest)
- `results/metrics/phase26_clean_compositional/` (verified: no predictions.npy; reconstructed predictions clearly labeled; original MISSING)
- `results/metrics/phase27_composition_diagnosis/` (verified: case E PARTIAL; aggregation gap negative)
- `results/metrics/phase28_oracle_complementarity/` (new artifacts; no synthetic content; approximate calculations clearly labeled)

## Key Findings
- Approximate oracle pair union suggests recoverable information may exist (independence assumption for binary classification).
- Aggregation gap remains negative (composition < single expert overall and RC).
- RC component conflict remains strongest established finding (C_change > R_change; RC composition < best single RC).
- Exact overlap/complementarity NOT computed (requires predictions.npy per architecture — MISSING for Phase 26 portfolio).
- Frozen linear probes NOT executed (requires checkpoint.pt evaluation).

## Artifacts (Verified Non-Empty)
- `aggregation_gap_detailed.csv` (1253 bytes)
- `approximate_complementarity.csv` (1107 bytes)
- `approximate_error_overlap.csv` (25471 bytes)
- `case.json` (2291 bytes)
- `expert_family_accuracy_summary.csv` (1325 bytes)
- `hypotheses.json` (1759 bytes)
- `intervention_gate.json` (1070 bytes)
- `manifest.json` (1718 bytes)
- `oracle_gap.csv` (1275 bytes)
- `oracle_union.csv` (32228 bytes)
- `prediction_reference.json` (877 bytes)
- `rc_component_sensitivity.csv` (365 bytes)
- `rc_disagreement_approximate.csv` (3354 bytes)
- `reproduction.json` (874 bytes)
- `summary.json` (2480 bytes)

## Limitations (Explicit, Not Synthetic)
- Phase 26 portfolio predictions.npy MISSING. Only Phase 25B MLP predictions.npy verified.
- All overlap/complementarity calculations approximate (independence assumption).
- Frozen linear representation probes deferred (not synthetic; requires checkpoint evaluation).
- Sequential order analysis deferred (Phase 26 fixed compositions only).
- No architecture change; RC remains CLOSED; intervention gate CLOSED.
