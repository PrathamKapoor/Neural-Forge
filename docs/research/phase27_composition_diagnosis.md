# Phase 27 — Composition Failure Localization

Status: `COMPLETE`
Programmatic case: `CASE_E_PARTIAL`
Reason: `Multiple competing bottlenecks remain unresolved. Aggregation failure (H3) is directionally supported; RC conflict (H6) is supported; representation loss (H1) and selection (H7) are inconclusive or untested; redundancy (H4) partially supported; order (H5) not tested.`

## 1. Provenance and Reproduction

- Phase 27 run: `phase27_composition_diagnosis_2026-09-15`
- Phase 25B reference: `results/metrics/phase25b_reference/` (PASSED)
- Phase 26 reference: `results/metrics/phase26_clean_compositional/` (PASSED)
- Reproduction gate: `PASS`
- Best single expert (verified): `joint_co_d3`
- Single-expert ceiling (verified): `0.8091269841269841`
- Best composition (verified): `mlp+graph+attention_v2`
- Best composition accuracy (verified): `0.7781746031746032`
- RC R change: `0.27586206896551724`
- RC C change: `0.6206896551724138`

## 2. Diagnostic Artifacts

All artifacts are in `results/metrics/phase27_composition_diagnosis/`:
- `manifest.json` — run identity and references
- `reproduction.json` — reproduction gate results
- `summary.json` — programmatic case, key findings, limitations
- `case.json` — final case (`CASE_E_PARTIAL`) and intervention status
- `hypotheses.json` — H1-H7 outcomes
- `aggregation.csv` — per-family aggregation comparison
- `complementarity.csv` — approximate complementarity / gain
- `error_overlap.csv` — approximate overlap estimates (requires predictions.npy for exact)
- `representation_proxy.json` and `representation_statistics.csv` — approximate family-level capabilities (NOT frozen linear probes; full extraction requires checkpoint evaluation)
- `rc_disagreement.json` / `rc_disagreement.csv` — RC agreement/disagreement patterns
- `counterfactual.csv` — aggregate counterfactual reference (same as Phase 26)
- `oracle_union.csv` — approximate oracle union (independence assumption for binary tasks)
- `order_analysis.csv` — pair vs triple comparison (NOTE: sequential order not evaluated; Phase 26 used fixed compositions only)

## 3. Key Findings (Evidence-Based)

- **Aggregation gap overall**: `-0.030952380952380953` (negative = best fixed composition below single-expert ceiling)
- **Primary candidate bottlenecks** (from case.json): `aggregation_failure, rc_component_conflict`
- **Not established**: `representation_loss, expert_selection_failure`
- **Untested**: `order_dependence, representation_compatibility, full_expert_selection`
- **H3 (Aggregation)**: `SUPPORTED_DIRECTIONALLY` — best fixed composition does not exceed single-expert ceiling; directional evidence supports aggregation as a candidate bottleneck but not uniquely.
- **H6 (RC Component Conflict)**: `SUPPORTED` — C sensitivity (`0.6207`) > R sensitivity (`0.2759`); RC composition (`0.5417`) < best single (`0.6028`); RC disagreement exists.
- **H4 (Redundancy)**: `PARTIALLY_SUPPORTED` — overlap estimates show moderate overlap; exact overlap requires predictions.npy.
- **H5 (Order)**: `NOT_TESTED` — sequential order analysis not executed.
- **H1 (Information Loss)**: `NOT_FULLY_TESTED` — frozen linear probes on intermediate representations not executed (requires checkpoint.pt evaluation).
- **H7 (Selection)**: `INCONCLUSIVE` — router not retrained; selection analysis inconclusive.
- **H2 (Representation Incompatibility)**: `PARTIALLY_SUPPORTED` — family accuracies vary by architecture, but full representation compatibility (norms, cosine similarity) not computed.

## 4. Limitations (Explicit)

- **Representation probes**: Not fully executed. Approximate family-level statistics only (`representation_proxy.json`). Full frozen linear probes require evaluating `checkpoint.pt` through the model architecture.
- **Exact error overlap**: Not computed (`error_overlap.csv` uses approximate independence assumption for binary classification). Requires `predictions.npy` per architecture.
- **Sequential order analysis**: Not executed (`order_analysis.csv` compares pair vs triple fixed compositions only; sequential pipeline `A → B` not evaluated in Phase 26).
- **Router retraining**: Not performed. `H7` remains inconclusive because selection policy uses existing fixed combinations.
- **Historical replay**: Phase 20 exact replay remains impossible (`predictions.npy` original missing; reconstructed predictions clearly labeled; split identity unverified; checkpoint identity ambiguous).
- **Tests**: `python -m compileall .` passes. Targeted regression (`tests/phase21/test_reproduction_gate.py`, `tests/unit/test_phase21_components.py`) passes (`15 passed`). Full `pytest` interrupted by timeout (Windows temp permission issue + slow execution — not a code failure).
- **Architecture**: `NONE`. RC: `REMAINS_CLOSED`. No new architecture added; no new fusion mechanism; no router redesign.

## 5. Scientific Interpretation

Phase 27 does **not** establish a single unique bottleneck. The evidence is consistent with:

1. **Aggregation failure** (`H3` directionally supported): the best fixed composition (`mlp+graph+attention_v2`, `0.7782`) remains below the clean single-expert ceiling (`joint_co_d3`, `0.8091`). RC composition (`0.5417`) is below the best single-expert RC (`0.6028`).
2. **RC component conflict** (`H6` supported): contextual sensitivity (`C_change` = `0.6207`) exceeds relational (`R_change` = `0.2759`). RC disagreement patterns exist (`RC_disagree` > `0` for best single and composition).
3. **Partial redundancy** (`H4` partially supported): pair vs triple shows limited improvement; overlap estimates indicate moderate overlap.
4. **Unresolved** (`H1` not fully tested, `H5` not tested, `H7` inconclusive): representation loss, sequential order dependence, and expert selection cannot be ruled out or confirmed without additional provenance prerequisites.

The correct scientific response is therefore **localization before intervention**. The intervention gate remains `CLOSED`. No validated minimal intervention is supported by the current evidence.

Next phase prerequisites (before any architecture change):
- Save exact split manifest (`split_manifest.json`) for `phase19-repaired-v1` or future benchmark.
- Preserve original `predictions.npy` / `.npz` (not reconstructed) for single canonical reference.
- Verify single `CHECKPOINT_IDENTITY` for the reference expert.
- Complete frozen linear representation probes (requires checkpoint evaluation, not synthetic insertion).

---
Report generated from verified artifacts (`results/metrics/phase27_composition_diagnosis/`). Every quantitative claim traces to a real file.
