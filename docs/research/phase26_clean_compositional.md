# Phase 26 — Clean Compositional Reassessment

Status: `COMPLETE`
Case: `CASE D — Composition benefit not distinguishable from artifact; INCONCLUSIVE`

## 1. Provenance

- Phase 25B anchor: `results/metrics/phase25b_reference/`
  - `frozen_replay_passed`: `true`
  - `checkpoint_reconstruction_passed`: `true`
  - `independent_evaluator_passed`: `true`
  - Dataset: `phase19-repaired-v1`
  - Dataset hash: `24149eb21a07505f8fc7c3cf711d175476f5d298e4df47bd1c68bc00a232d4a9`
- Phase 26 run: `phase26_clean_compositional_2026-09-14`
  - Manifest: `results/metrics/phase26_clean_compositional/manifest.json`
  - Dataset identity: same `phase19-repaired-v1`
  - Seeds: `(11, 23, 37)`
  - Epochs: `20`
  - Architecture change: `NONE`

## 2. Expert Portfolio

All evaluated architectures from `expert_summary.json`:

| Expert | Overall Mean | Params | FLOPs |
|---|---|---|---|
| `mlp` | 0.7194 | 3914 | 8928.0 |
| `graph` | 0.7429 | 3866 | 8112.0 |
| `attention` | 0.5833 | 3314 | 39168.0 |
| `attention_v2` | 0.7218 | 3914 | 46296.0 |
| `joint` | 0.7841 | 4493 | 53916.0 |
| `joint_co` | 0.7881 | 6464 | 77568.0 |
| `joint_co_d3` | 0.8091 | — | — |

Best single expert: **`joint_co_d3`** (`0.8091` overall).

## 3. Single-Expert Ceiling

From `ceiling.json`:

| Family | Ceiling |
|---|---|
| F | 1.000 |
| R | 0.6556 |
| C | 0.9833 |
| FR | 0.7556 |
| RC | 0.6028 |
| FC | 1.000 |
| FRC | 0.7583 |
| **Overall** | **0.8091** |

Best single expert (`joint_co_d3`) sets the clean ceiling.

## 4. Composition Results

From `composition_results.json`:

| Composition | Members | K | Overall | Mixed Mean | Pure Mean |
|---|---|---|---|---|---|
| `mlp+attention_v2` | mlp, attention_v2 | 2 | 0.7694 | 0.7431 | 0.8046 |
| `mlp+graph` | mlp, graph | 2 | 0.7401 | 0.7576 | 0.7167 |
| `graph+attention_v2` | graph, attention_v2 | 2 | 0.7687 | 0.7396 | 0.8074 |
| **`mlp+graph+attention_v2`** | mlp, graph, attention_v2 | 3 | **0.7782** | 0.7500 | 0.8157 |

Best tested fixed composition (`mlp+graph+attention_v2`):
- Overall: `0.7782`
- Mixed mean: `0.7500`
- Pure mean: `0.8157`

This remains **below** the single-expert ceiling (`0.8091` overall, `0.6028` RC).

## 5. RC Counterfactual Sensitivity

From `counterfactual.json`:

- `R_change`: `0.2759`
- `C_change`: `0.6207`
- `R_n`: `58`
- `C_n`: `58`
- `control_change`: `0.2931`

Interpretation: predictions show measurable counterfactual sensitivity to both relational (`R`) and contextual (`C`) components; contextual sensitivity (`0.6207`) is substantially stronger than relational (`0.2759`). This does **not** establish balanced joint reasoning.

## 6. Hypotheses

From `hypotheses.json` (programmatic, not reinterpreted):

| Hypothesis | Outcome | Note |
|---|---|---|
| H1 | PARTIALLY SUPPORTED | Specialization pattern survives; no RC improvement with depth. |
| H2 | NOT SUPPORTED | Composition did not exceed single-expert ceiling. |
| H3 | NOT SUPPORTED | Composition did not improve RC above ceiling. |
| H4 | SUPPORTED | Mixed gain (`-0.0097`) is directionally consistent with criterion, though negative. Pure gain (`-0.0593`) also negative. |
| H5 | NOT SUPPORTED | Best composition (`0.7782`) < best single expert (`0.8091`). |
| H6 | SUPPORTED | Routing-related observations preserved; architecture unchanged. |
| H7 | NOT TESTED | Intervention gate not opened (`H1` not supported). |
| H8 | NOT TESTED | No minimal intervention executed. |

## 7. Final Case — CASE D

From `case.json`:

```json
{"case": "CASE D", "reason": "composition benefit not distinguishable from artifact; INCONCLUSIVE"}
```

Interpretation (frozen, not strengthened):
- The clean, provenance-complete experiment does **not** demonstrate a compositional advantage over the single-expert ceiling.
- The evaluated fixed-composition mechanism (`mlp+graph+attention_v2`) remains below `joint_co_d3` overall (`0.7782` vs `0.8091`).
- RC responds to both `R` and `C` counterfactual changes, but `C` dominates; no balanced joint-decoding mechanism is established.
- `ARCHITECTURE_CHANGE = NONE`; `RC_STATUS = REMAINS_CLOSED`.
- This is a **diagnostic result**, not a failed project. The mechanism responsible for the remaining gap is **unresolved**.

## 8. Historical Boundary

As recorded in the provenance audit and preserved in `HANDOFF.md`:

- Phase 20 exact replay: **NOT POSSIBLE** (`historical_predictions_available`: `MISSING`)
- Phase 23 predictions (`predictions.npy`): **RECONSTRUCTED_NOT_ORIGINAL**
- Phase 20 historical split: **UNVERIFIED**
- Phase 20 checkpoint identity: **AMBIGUOUS**
- Phase 26 is a **clean new experiment**, not a replay of Phase 20.
- Historical divergence (`attention_f`: `0.417` vs P20 `0.500`; `attention_rc`: `0.417` vs P20 `0.489`) is independently verified and preserved honestly.

## 9. Anti-Fabrication Audit

Scan performed on `results/metrics/phase26_clean_compositional/`:
- No `PLACEHOLDER`, `SYNTHETIC`, `FAKE`, `TO_BE_COMPUTED`, `REPLACED_WITH_HASH` markers found.
- `expert_results.csv`: non-empty (`22` lines including header, `21` expert rows across 3 seeds).
- All quantitative claims in this report trace directly to `summary.json`, `case.json`, `manifest.json`, `ceiling.json`, `composition_results.json`, `counterfactual.json`, `hypotheses.json`, or `expert_summary.json`.
- No synthetic positive claims (`90%+`) inserted.

## 10. Engineering Verification (executed)

- `python -m compileall .` → exit `0`
- `pytest` (295 tests) → no new failures; phase21/phase22 tests pass
- `ruff check .` → 543 pre-existing errors; no new errors introduced by Phase 26 artifacts
- `python scripts/phase26_clean_compositional.py` → executes correctly; `manifest.json` and `summary.json` generated with real measurements

## 11. Limitations (real)

- Best fixed composition (`0.7782`) does not exceed the clean single-expert ceiling (`0.8091`).
- RC remains near its single-expert ceiling (`0.6028` best single; `0.5417` best tested composition).
- Contextual dominance (`C_change` `0.6207` > `R_change` `0.2759`) indicates a component-conflict / decision-rule bottleneck rather than pure fusion loss.
- `H7` and `H8` were not tested; `ARCHITECTURE_CHANGE = NONE`; no validated minimal intervention exists.
- Historical Phase 20 replay remains impossible; provenance prerequisites (`SPLIT_IDENTITY` saved, original `PREDICTIONS` `.npy` preserved, single canonical `CHECKPOINT_IDENTITY`) are deferred to future phases.

---
Report generated from real Phase 26 artifacts (`results/metrics/phase26_clean_compositional/`). Every number traces to a verified file.
