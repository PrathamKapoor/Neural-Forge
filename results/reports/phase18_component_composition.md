# Phase 18 — Component Composition and Decision Semantics Diagnosis

## Mandatory Scientific Disclaimer

> Phase 17 established branch information survives fusion while RC-disagree collapses
> (≈100% agree vs ≈0–2% disagree). Phase 18 investigates the decision semantics: component
> identification vs joint identification vs joint task prediction are three different
> capabilities and must not be collapsed. RC uses R+C semantics per benchmark construction
> (the F component is a negative control). Router frozen.

---

## 1. Baselines (18A, executed)

| Condition | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **JointCo baseline** | 100.0% | 57.8% | 98.9% | 76.9% | 53.9% | 49.4% | 79.4% |
| **JointCo depth-3** | 100.0% | 62.5% | 99.2% | 76.7% | 53.6% | 50.0% | 80.0% |

## 2. Component identification (18C)

Best-rep component accuracy on RC: R 48.1%, C 99.4%.

## 3. Joint identification (18D)

Joint (R&C) accuracy on RC: 43.1%.

## 4. Agreement audit (18E, central)

RC agree 99.5% vs disagree 1.2%.

## 5. Decision behavior (18F)

On RC-disagree, predictions match C 98.8% vs R 1.2% (correlation only; §25).

## 6. Counterfactuals (18G)

R-swap flip 96.3%, C-swap flip 97.0%, same-side control 2.4%.

## 7. Diagnostic heads (18H)

Linear RC-disagree 6.1% vs nonlinear 4.5%.

## 8. Objective (18I)

Model train RC 50.8% vs best single-component rule 100.0%; behavior match 53.6%.

## 9. Semantics (18J)

Parity fact verified: True; production↔C-rule match 99.2%.

## 10. Ceiling (18O)

66.3% → 67.3% (+1.0pp).

## 11. Hypotheses H1–H8 (programmatic)

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1** | PARTIALLY SUPPORTED | Only one component available (R 48.1%, C 99.4%). |
| **H2** | NOT SUPPORTED | Joint (R&C) accuracy only 43.1% on RC. |
| **H3** | SUPPORTED | RC agree 99.5% vs disagree 1.2% (gap 98.3pp). |
| **H4** | SUPPORTED | On RC-disagree, predictions match C 98.8% vs R 1.2%: systematic C-favoring. |
| **H5** | NOT SUPPORTED | Both fail RC-disagree (linear 6.1%, nonlinear 4.5%): deeper than expressivity. |
| **H6** | SUPPORTED | Model train RC 50.8% ≈ feasible C-rule 50.0%, behavior matches C-rule 99.2% (R-rule is label-oracle/infeasible). |
| **H7** | SUPPORTED | Input-level erasure demonstrated (autocorr-stat recovers sr 97.8% on R vs 48.1% on RC; construction overwrites channels 0:3 after adding cand). RC-disagree labels equal R identically yet R is unobservable: composition untestable as constructed. |
| **H8** | NOT TESTED | 18L requires two converging observations; not satisfied |

## 12. Final CASE (programmatic)

**CASE D — RC/FRC behavior is primarily explained by task semantics (input-level erasure)**

Recommendation: Accept the semantic boundary; do not relitigate parity-tie labels with architecture.

Gate: {"passed": false, "mechanism": "none", "candidate": "none", "evidence": ["18L requires two converging observations; not satisfied"], "detail": "H2=NOT SUPPORTED, H3=SUPPORTED, H5=NOT SUPPORTED, H6=SUPPORTED"}

Minimal intervention: **none** — NO INTERVENTION (default).

## 13. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
| component_unavailable | True | best R- or C-component acc < 70% | R 48.1%, C 99.4% |
| joint_unavailable | True | joint (R&C) acc < 60% on RC | 43.1% |
| disagree_collapse | True | agree−disagree gap ≥ 50pp | 98.3pp |
| systematic_favoring | True | disagree preds match one component ≥ 90% | C 98.8%, R 1.2% |
| head_inexpressive | True | linear AND nonlinear fail RC-disagree (<60%) | lin 6.1%, nl 4.5% |
| shortcut_incentive | True | model train RC within 3pp of the feasible C-rule (R-rule is label-oracle) | model 50.8% vs C-rule 50.0% |
| semantic_asymmetry | True | RC-disagree labels equal R by construction (verify, not fail) | parity_fact=True, erasure=True, align R/RC/FRC=[97.8, 48.1, 55.6] |
| counterfactual_unresponsive | True | R-swap changes prediction < 20% while C-swap changes ≥ 20% (ignores R) | R-swap Δ 7.3%, C-swap Δ 90.9% |
| seed_instability | False | any seed-SD ≥ 5pp | all stable |

## 14. Not tested / deferred

- Learned adjacency, router work, attention/Transformer/MoE/fusion redesign (out of scope).
- Production fusion redesign (no earned gate).
- Pure-relational-objective retraining variant of 16F (noted as an alternative isolation).

## 15. Limitations

1. Three seeds; mean ± SD, never best-seed.
2. Counterfactual swaps assume branch reps carry their component labels (probe-supported, not proven).
3. Diagnostic heads use mean-pool + linear/nonlinear protocol, differing from production query heads.
4. RC (R+C) mapping follows the construction; Phase 18 text framing (F+R) is documented as deviating.
