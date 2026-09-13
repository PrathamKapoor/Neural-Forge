# Phase 21 — RC Relational Capacity and Decision Conversion Diagnosis

## Executive summary

Programmatic verdict: **CASE E — Decision-rule / component-conflict bottleneck**.

## Phase 20 reproduction

Gate: PASS.
Reproduction compares recomputed Phase 20 values (same weights, same seeds) within 3pp per family.

## Research question

Why does the existing prediction mechanism select one available component when RC requires combining R+C?

## Pre-registered hypotheses

H1 depth limitation; H2 task alignment; H3 contextual dominance; H4 combination
causality; H5 head expressivity; H6 locality; H7 order causality; H8 gated capacity
intervention. Thresholds in `evaluation/phase21_rc_diagnosis.py`.

## Experimental design

Depth-1/2/3 (+gated depth-4) trained on repaired F+R+C with the exact Phase 20
joint protocol; everything else frozen diagnostics on depth-3 (and cheap variants).

## RC observability

Repaired RC relational carrier observable per Phase 19 gate (rechecked in 20A smoke).

## Relational capacity analysis (H1)

Per-depth RC (mean): depth1=53.1%, depth2=53.1%, depth3=52.8%.
Gains vs depth-1: [0.0, 0.0, 0.0].

## Representation probes (H2)

Best-location R decodability 64.8% vs final RC-target decodability 52.8%.

## R/C contribution analysis (H3)

Mean grad shares: R 9.3%, C 23.3% (C/R ratio 2.5x).

## RC agreement/disagreement

Depth-3 agree 98.0% vs disagree 1.3%.

## Counterfactual decision sensitivity (H4 + §12)

R change 0.0% (correct 0.0%); control change 1.3%.

## Composition-order analysis (H7)

Controlled-readout RC spread +1.1pp.

## Minimal intervention (H8, only if gated)

Gate: {"passed": false, "reason": "H1=NOT SUPPORTED (need SUPPORTED); R share 9.3% (need >= 5%)."}.

## Compute/latency

Per-depth params, relational params, latency in `compute.csv`; intervention candidate costs in `intervention.csv` when tested.

## Seed statistics

3/3 directional consistency required for small effects; per-seed values in `seed_results.csv`.

## H1–H8 verdict table

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1** | NOT SUPPORTED | No specific RC improvement with depth (+0.0pp, 0/3 seeds positive). |
| **H2** | NOT SUPPORTED | R itself poorly decodable (64.8%): information missing, not conversion. |
| **H3** | PARTIALLY SUPPORTED | C/R ratio 2.5x, R share 9.3% (continuous; weak dominance signal). |
| **H4** | NOT SUPPORTED | R counterfactuals rarely change the decision (0.0% vs control 1.3%): conversion/use failure. |
| **H5** | NOT SUPPORTED | Frozen heads fail RC (best 55.0%): problem precedes the head. |
| **H6** | NOT SUPPORTED | Relational output lacks R (64.8%). |
| **H7** | NOT SUPPORTED | Controlled order spread only +1.1pp on RC. |
| **H8** | NOT TESTED | H8 gate did not pass; candidate not trained. |

## CASE A–F programmatic verdict

**CASE E — Decision-rule / component-conflict bottleneck**

## What this rules out

Recorded in `failure_diagnosis.csv` with explicit criteria.

## What remains unresolved

See recommendation: Make R/C-disagreement resolution the primary Phase 22 target.

## Phase 22 recommendation

Make R/C-disagreement resolution the primary Phase 22 target. Minimal intervention: **none** — No architectural intervention.
