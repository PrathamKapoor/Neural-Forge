# Phase 22 — Experimental Reproducibility and Provenance Integrity Audit

Status: Provenance audit complete; frozen replay confirms divergence at metric/evaluation stage; reproduction gate (Phase 21) fails closed.

## Protocol enforcement (§2, §22A-K)

No new architecture, no RC improvement, no re-run until divergence identified.

## Provenance audit (22A)

Artifact identity recorded in `scripts/phase22_audit_provenance.py` (`provenance_manifest`). Key artifacts preserved:
- `docs/architecture/phase20_log.md`
- `results/reports/phase20_repaired_portfolio.md`
- `results/metrics/phase20_repaired_portfolio/summary.json`
- `results/metrics/phase21_rc_diagnosis/summary.json` (existing broken artifacts preserved)
- `src/neuroforge/evaluation/phase21_rc_diagnosis.py`
- `src/neuroforge/training/phase21_rc_diagnosis.py`

## Dataset identity (22B-22C)

Phase 20 uses repaired benchmark `phase19-repaired-v1` (C→F→R(ch5)). Phase 21 uses same benchmark per manifest. No separate dataset divergence confirmed at construction level; divergence occurs at measurement stage (0.0 values from broken evaluation pipeline).

## Frozen-artifact replay (22H — critical test)

Replays Phase 20 predictions through Phase 21 evaluator:
- Phase 20 predictions/artifacts exist.
- Phase 21 evaluator outputs diverged results (many 0.0/MISSING for RC metrics).
- Divergence confirmed at evaluation/metric stage, not at dataset construction or training stage.

Outcome: Outcome 2 (evaluator diverges) or Outcome 3 (artifacts incomplete). Given broken measurements (agreement 0.0% vs P20 98.0%), divergence is at evaluation stage.

## Programmatic verdict (§22L)

CASE: E / G hybrid — reproduction divergence at metric/evaluation stage; provenance completeness audit required.

Evidence: P20 predictions preserved; P21 measurements broken (many 0.0/MISSING). Divergence flags show 0 metrics passing; all RC metrics diverged. Frozen replay confirms divergence at evaluation stage.

What remains unresolved:
- Fix measurement/evaluation pipeline before reopening RC diagnosis.
- Once pipeline fixed, rerun frozen replay; if replay then passes, proceed to H1-H8 (capacity, representation, combination, decision conversion) per Phase 21 protocol.
- If replay still fails after pipeline fix, audit dataset/split/checkpoint divergence (CASE B/C/D).

No architecture change recommended (§20, §22 protocol).
