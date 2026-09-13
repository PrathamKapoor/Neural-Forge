# Phase 25B — Reference Lineage (research log)

## Purpose

Establish the first clean, authoritative, replayable reference run after the
Phase 21–24 forensic failure. This is lineage infrastructure, not a scientific
experiment.

## Protocol

- Dataset: `phase19-repaired-v1` (unchanged).
- Split: deterministic, seed 11 (train) / 11+50000 (test); persisted to
  `split_manifest.json` with actual sample IDs.
- Model: existing MLP (`input_dim=8, hidden_dim=24, depth=3, num_classes=2`).
  No tuning, no new architecture.
- Seed: 11. Optimizer AdamW, lr 0.003, weight decay 1e-4, batch 60, 2 epochs.
- Evaluator: canonical `compute_metrics`; independent evaluator is a separate
  numpy-path implementation.

## Results

Overall accuracy 0.7167; per family F 1.0 / R 0.4917 / C 0.5 / FR 0.775 /
RC 0.55 / FC 0.9833 / FRC 0.7167. RC 62 agree / 58 disagree.

Frozen replay PASSED (diff 0.0), checkpoint reconstruction PASSED (identical
hash), independent evaluator AGREES (diff 0.0).

## Decisions

- `REFERENCE_LINEAGE = ESTABLISHED` (MLP authoritative).
- `ARCHITECTURE_CHANGE = NONE`.
- `RC_STATUS = REMAINS_CLOSED`.
- `PHASE_26_GATE = OPEN` for the clean compositional reassessment, which may
  now be planned against this reference rather than the provenance-incomplete
  historical Phase 20.

## Deviations

None. The prior syntactically-broken `scripts/phase25b_reference_lineage.py`
was replaced with a correct implementation that actually executes and produces
real artifacts.