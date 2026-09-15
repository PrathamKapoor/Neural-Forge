# Phase 29 — Prediction Recovery & Exact Complementarity

Status: `COMPLETE` | Case: `CASE_D_UNRESOLVED` | Architecture: `NONE` | RC: `CLOSED` | Intervention: `NONE` | Gate: `CLOSED`

## Evidence Sources
- `results/metrics/phase25b_reference/` (verified: MLP predictions.npy, checkpoint.pt, manifest)
- `results/metrics/phase26_clean_compositional/` (verified: predictions.npy MISSING; reconstructed clearly labeled; original MISSING)
- `results/metrics/phase29_prediction_recovery/` (new artifacts: real, non-empty, no synthetic markers)

## Key Findings
- Phase 25B MLP predictions.npy verified (hash: cf6afa609...).
- Phase 26 portfolio predictions.npy MISSING (historical provenance issue).
- Exact overlap/complementarity requires predictions.npy per architecture (MISSING for portfolio experts).
- Approximate analysis available from family accuracies (independence assumption).
- No predictions fabricated; no reconstructed predictions substituted.
- Historical Phase 20 replay impossible.

## Artifacts (Non-Empty)
- `approximate_vs_exact.csv` (629 bytes)
- `case.json` (1320 bytes)
- `checkpoint_inventory.csv` (1003 bytes)
- `hypotheses.json` (2031 bytes)
- `manifest.json` (997 bytes)
- `prediction_reference.json` (1952 bytes)
- `reconstruction.json` (2373 bytes)
