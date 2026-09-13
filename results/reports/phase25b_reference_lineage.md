# Phase 25B — Clean MLP Reference Lineage

## What this run is

A **new, authoritative MLP reference lineage**, freshly executed end-to-end in
the real workspace. It establishes a single clean reference run (dataset →
split → configuration → seed → MLP → checkpoint → predictions → metrics →
frozen replay → checkpoint reconstruction → independent evaluator) so that
future scientific phases have a provenance-complete baseline.

## What this run is NOT

- **NOT** historical Phase 20 replay. Phase 20 remains provenance-incomplete.
- **NOT** an RC optimization. No architecture change, no router change, no
  benchmark change, no new expert.
- **NOT** Phase 26. The compositional reassessment remains blocked until this
  reference lineage is genuinely established.

## Historical P20 status (unchanged)

- original Phase 20 predictions: **MISSING**
- historical split: **UNRESOLVED / UNVERIFIED**
- historical checkpoint identity: **AMBIGUOUS** (portfolio of `.pt` files)
- reconstructed `predictions.npy` (Phase 23): remains
  `RECONSTRUCTED_NOT_ORIGINAL`; never promoted to historical original.

## New MLP reference status (from actual artifacts only)

Run ID: `phase25b_reference_2026-09-14`
Dataset: `phase19-repaired-v1` (verified identical to validated benchmark)
Split: persisted `split_manifest.json` (train seed 11, test seed 11+50000; actual sample IDs + hashes)
Model: MLP — `input_dim=8, hidden_dim=24, depth=3, num_classes=2` (validated existing architecture, no tuning)
Seed: 11 (single clean reference seed)
Epochs: 2 (lineage verification; not an optimization run)

### Results (measured from persisted predictions)

| Gate | Result |
|---|---|
| Overall accuracy | 0.7167 |
| F | 1.0000 |
| R | 0.4917 |
| C | 0.5000 |
| FR | 0.7750 |
| RC | 0.5500 |
| FC | 0.9833 |
| FRC | 0.7167 |

RC: 120 samples — 62 agree (acc 0.5484), 58 disagree (acc 0.5517).

### Provenance gates

| Gate | Status |
|---|---|
| Frozen replay (predictions → metrics, no checkpoint loaded) | **PASSED** (diff 0.0) |
| Checkpoint reconstruction (.pt → predictions, hash match) | **PASSED** (identical) |
| Independent evaluator (separate numpy path) | **AGREES** (diff 0.0) |

All three pass → the MLP reference is **AUTHORITATIVE**.

## Artifacts

`results/metrics/phase25b_reference/`:

- `config.json` (hash `b6b686d2…`)
- `split_manifest.json` (hash `e42013cc…`)
- `dataset_identity.json` (hash `24149eb2…`)
- `checkpoint.pt` (hash `d8d6613a…`) + `checkpoint.json`
- `predictions.npy` (hash `cf6afa60…`) + `labels.npy` + `sample_ids.npy`
- `metrics.json`
- `frozen_replay.json`, `checkpoint_reconstruction.json`, `independent_evaluation.json`
- `manifest.json` + `provenance_manifest.json` + `provenance_summary.json` + `reference_status.json`

Anti-fabrication scan: no `PLACEHOLDER`, `TO_BE_COMPUTED`, `REPLACED_WITH_HASH`,
`SYNTHETIC`, or `FAKE` present.

## Historical boundary (preserved)

Phase 21 CASE E, Phase 23 CASE G PARTIAL, ARCHITECTURE = NONE, RC = CLOSED are
preserved. This Phase 25B MLP run is a separate, new lineage; it does not claim
or refute the historical Phase 20 result and does not use reconstructed Phase 23
predictions as ground truth.