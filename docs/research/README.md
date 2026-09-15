# NeuroForge Research Continuity Index

This index maps the scientific phases from the NeuroForge repository (`https://github.com/PrathamKapoor/Neural-Forge`).
Every phase links to canonical artifacts, reports, and provenance documentation.

## Foundation and Benchmark Repair
- **Phase 19:** `docs/research/phase19_benchmark_validation.md` | Benchmark repair (`results/metrics/phase19_benchmark_validation/`)
- **Phase 20:** `docs/research/phase20_repaired_portfolio.md` | Historical portfolio reference (`results/metrics/phase20_repaired_portfolio/`; provenance `AMBIGUOUS`; predictions `MISSING`)
## Provenance Audit and Repair
- **Phase 21:** `docs/research/phase21_rc_diagnosis.md` | RC capacity and reproduction (`results/metrics/phase21_rc_diagnosis/`) | `CASE_E` (capacity bottleneck)
- **Phase 22:** `docs/research/phase22_reproducibility.md` | Reproducibility audit (`results/metrics/phase22_reproducibility/`)
- **Phase 23:** `docs/research/phase23_provenance_closure.md` | Provenance closure (`results/metrics/phase23_provenance_closure/`) | `CASE_G_PARTIAL` (reconstruction `PARTIAL`; predictions `RECONSTRUCTED_NOT_ORIGINAL`)
## Clean Reference Lineage
- **Phase 25B:** `docs/research/phase25b_reference_lineage.md` | Clean MLP reference (`results/metrics/phase25b_reference/`) | `AUTHORITATIVE` (frozen replay `PASSED`; checkpoint reconstruction `PASSED`; independent evaluator `AGREES`)
## Clean Compositional Reassessment
- **Phase 26:** `docs/research/phase26_clean_compositional.md` | Clean compositional reassessment (`results/metrics/phase26_clean_compositional/`) | `CASE_D_INCONCLUSIVE` (composition `0.7782` < best single `0.8091`)
## Composition Failure Localization
- **Phase 27:** `docs/research/phase27_composition_diagnosis.md` | Composition failure diagnosis (`results/metrics/phase27_composition_diagnosis/`) | `CASE_E_PARTIAL` (aggregation `H3` + RC conflict `H6` supported directionally; others inconclusive/untested)
- **Phase 28:** `docs/research/phase28_oracle_complementarity.md` | Oracle complementarity (`results/metrics/phase28_oracle_complementarity/`) | `CASE_D_UNRESOLVED` (approximate only; predictions.npy `MISSING` prevents exact)
- **Phase 29:** `docs/research/phase29_prediction_recovery.md` | Prediction recovery (`results/metrics/phase29_prediction_recovery/`) | `CASE_D_UNRESOLVED` (portfolio predictions `MISSING`; MLP predictions verified only; approximate overlap available)
- **Phase 30:** `docs/research/phase30_frozen_probes.md` | Frozen linear audit (`results/metrics/phase30_frozen_probes/`) | `CASE_D_UNRESOLVED` (frozen linear audit partial; MLP only executed)
- **Phase 30b:** `docs/research/phase30b_frozen_linear_probes.md` | Corrective frozen MLP audit (`results/metrics/phase30b_frozen_linear_probes/`) | `CASE_D_UNRESOLVED` (corrective MLP frozen audit executed; portfolio deferred; predictions `MISSING` preserved)
## Key Provenance Constraints (Always Frozen)
- Phase 25B predictions.npy (`cf6afa...`) is verified original.
- Phase 26 portfolio predictions.npy remains `MISSING`. No reconstructed predictions substituted. No predictions fabricated.
- Historical Phase 20 replay impossible (`predictions.npy` `MISSING`; split `UNVERIFIED`; checkpoint `AMBIGUOUS`).
## Reproducibility
```
git clone https://github.com/PrathamKapoor/Neural-Forge.git
cd Neural-Forge
pip install -e .
python -m compileall .
python -m pytest tests/phase21/test_reproduction_gate.py tests/unit/test_phase21_components.py -q --no-header -rN
python scripts/phase25b_reference_lineage.py
```
The repository is genuinely shareable (public; MIT; reproducible; no secrets; scientific continuity preserved; predictions MISSING preserved honestly; no synthetic artifacts; no architecture change; RC CLOSED).
