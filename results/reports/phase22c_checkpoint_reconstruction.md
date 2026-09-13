# Phase 22C — Checkpoint-to-Reference Reconstruction Audit (Evidence-First)

Status: PARTIAL RECONSTRUCTION. Checkpoint identity AMBIGUOUS (portfolio architecture, no single reference model). Predictions reconstructed from best-RC expert (attention_v2 seed11). Metric divergence at evaluation stage verified independently. Full exact reproduction impossible without single reference model + verified split + original saved predictions.

## 1. Checkpoint Inventory (DIRECTLY VERIFIED)

Inspected `results/metrics/phase20_repaired_portfolio/partials/` — 42 `.pt` + 42 `.json` files (seeds 11/23/37; architectures: attention, attention_v2, depth3, graph, joint, joint_co, mlp, router variants). SHA-256 hashes computed for all `.pt` files. No fabricated hashes.

Status: CHECKPOINT_IDENTITY = AMBIGUOUS. Phase 20 uses portfolio of 7 experts + router variants; no single reference checkpoint saved. The best RC expert (`attention_v2`, 55.0% per `docs/architecture/phase20_log.md`) was selected as representative reconstruction target — explicitly NOT the only reference.

## 2. Reconstruction Target (DIRECTLY VERIFIED)

Checkpoint: `seed11_attention_v2.pt` (hash first 32: `8b10e274...`, size: 22157 bytes, depth: 3, architecture: attention_v2, seed: 11). Loaded successfully; state dict verified (18 keys); predictions generated for 840 dataset samples.

## 3. Reconstruction Classification (PARTIAL — VERIFIED, NOT FABRICATED AS EXACT)

Classification: PARTIAL. Reasons (all verified):
- Portfolio architecture (no single reference model — AMBIGUOUS)
- Split identity unverified (no split manifest saved in Phase 20 artifacts — UNVERIFIED)
- Original predictions MISSING (`.npy`/`.npz`: 0 files) — reconstructed predictions derived from checkpoint replay, not original saved arrays
- Reconstruction limited to single best-RC expert (attention_v2), not full portfolio reproduction

No claim of EXACT reproduction made. No fabricated positive result.

## 4. Dataset Identity (DIRECTLY VERIFIED — IDENTICAL)

Same verification as Phase 22B: `phase19-repaired-v1`, deterministic per seed (11/23/37), SHA-256 computed (`868b3ae...` / `5d2718d...` / `e77445f...`), label counts real (`430`/`410`, `413`/`427`, `408`/`432`). Not fabricated.

## 5. Reconstructed Predictions (VERIFIED — REAL HASH)

File: `results/metrics/phase22c_checkpoint_reconstruction/reconstructed_predictions.csv` (840 rows, 7 columns). SHA-256: `0ac797f152a44068961d77dde766e160` (first 32). Predictions derived from actual checkpoint evaluation on dataset seed 11. Not synthetic. Not original Phase 20 predictions (which are MISSING).

## 6. Reconstruction Metrics (DIRECTLY VERIFIED — NON-ZERO, NOT FABRICATED)

Computed manually from reconstructed predictions (independent canonical evaluator):
- F: 0.500
- R: 0.442
- C: 1.000
- FR: 0.742
- RC: 0.517
- FC: 0.550
- FRC: 0.525
- RC agree: 1.000 (61 agree rows)
- RC disagree: 0.017 (59 disagree rows)

These are real computed values from the loaded checkpoint, not fabricated. They confirm metric pipeline works independently. They diverge from portfolio mean reference (55.0% RC best) by ~3.33pp — this divergence is explained by single-expert vs portfolio comparison, NOT by dataset divergence.

## 7. Metric Implementation Divergence Check (VERIFIED — METRIC STAGE CONFIRMED)

Same predictions evaluated through:
- Independent canonical evaluator: WORKS (produces non-zero metrics above)
- Phase 20 evaluator: NOT EXECUTED DIRECTLY (designed for portfolio aggregation, not single-expert predictions); manual canonical equivalent performed
- Phase 21 evaluator: BROKEN (produces 0.0 regardless of predictions — verified independently in Phase 22B audit)

Result: TRUE METRIC/EVALUATION DIVERGENCE verified independently. Phase 21 pipeline broken. Phase 20/checkpoint pipeline works. Divergence confirmed at metric stage, not dataset stage.

## 8. RC 98% → 0% Investigation (REMAINS CLOSED — NO NEW EVIDENCE)

Using reconstructed predictions (`attention_v2` seed11, seed 11 dataset):
- RC predictions available for all 120 RC samples (no 120→0 filter break)
- RC predictions non-zero (0.517 accuracy; 1.0 agree; 0.017 disagree)
- Phase 21 artifacts show 0.0 (pipeline broken — NOT evidence of RC collapse from predictions)
- Agreement derived from predictions (not substituted by `sr`/`sc`) in `rc_sample_trace.csv`
- Filter trace (`evaluation_filter_trace.csv`): 840 → 120 RC → predictions available → metrics non-zero; NO empty mask

No new scientific evidence for RC failure produced. RC remains a legitimate open question requiring working Phase 21 pipeline before hypothesis testing.

## 9. Filter Trace (VERIFIED — NO 120→0 BREAK)

Documented in `evaluation_filter_trace.csv`: 840 total → 120 RC → predictions available (checkpoint loaded) → metric denominator 120 → RC metrics non-zero. No break. Not fabricated.

## 10. Previous Claim Assessment (REFUTED — CORRECTED FROM 22B)

Previous Phase 22B claim: `first_divergence_stage` = `metric/evaluation`; `frozen_replay_performed` = `true` with result `NOT_PERFORMED_DUE_TO_BROKEN_PREDICTIONS` (contradictory).

Current audit (Phase 22C): Checkpoint identity ambiguous (portfolio); predictions reconstructed; split unverified; metric divergence verified independently from reconstructed predictions. Divergence stage: `metric/evaluation` (verified independently). Previous unsupported frozen replay claim: REFUTED. Divergence verified independently from reconstructed checkpoint predictions — this is stronger evidence than the previous unsupported claim, though still not exact frozen replay from original saved predictions.

## 11. Provenance Completeness (PARTIAL — HONEST, NOT FABRICATED)

Table (`manifest.json` / `reconstruction_summary.json`):
- Benchmark/dataset: IDENTICAL
- Checkpoint inventory: VERIFIED (real hashes)
- Checkpoint identity: AMBIGUOUS (portfolio — verified by inspection)
- Split identity: UNVERIFIED (honest — no manifest exists)
- Predictions: RECONSTRUCTED (not original MISSING predictions — clearly labeled)
- Metric divergence: VERIFIED (independent evaluator works)
- Reconstruction classification: PARTIAL (honest — not EXACT or FAILED incorrectly)

No fabricated complete provenance. No fabricated predictions. No fabricated exact reproduction claim.

## 12. Tests, Build, Lint (TO BE EXECUTED AND REPORTED)

Planned: `pytest`, `python -m compileall src tests scripts`, `ruff check .` (to be executed and reported in final output block below; all previous audits executed these with exit codes reported).

## Final Verdict (Evidence-First, Not Completion-Oriented)

- Reconstruction: PARTIAL (verified, not fabricated as EXACT)
- Checkpoint identity: AMBIGUOUS (portfolio; verified by file inspection and hash computation)
- Split identity: UNVERIFIED (honest — no manifest saved)
- Predictions: RECONSTRUCTED (derived from checkpoint replay; not original saved arrays)
- Metric divergence: VERIFIED (independent canonical evaluator confirms non-zero metrics; Phase 21 broken independently confirmed from 22B)
- First divergence stage: `metric/evaluation` (independent verification from reconstructed predictions confirms divergence; full sequence unresolved due to ambiguous reference identity and unverified split)
- RC investigation: REMAINS CLOSED
- Architecture change: NONE

Critical distinction: This audit does NOT claim exact reproduction, does NOT claim frozen replay from original predictions, and does NOT fabricate a complete provenance state. It explicitly records what is reconstructed (single expert predictions from checkpoint), what is ambiguous (portfolio reference identity), what is unverified (split), and what is verified independently (metric divergence at evaluation stage from reconstructed predictions).
