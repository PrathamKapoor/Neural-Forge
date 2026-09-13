# Phase 20 Research Log — Clean Portfolio Re-evaluation on Repaired Mixed Benchmark

## Starting evidence (from Phase 19)

- Repaired construction `phase19-repaired-v1` (C→F→R(ch5)); 8/8 validity gates.
- Historical RC/FRC composition results invalid for erased-signal claims; retained untouched.

## Protocol (no architecture changes)

- Fresh training on repaired data via protocol-mirrors (same optimizer/loss/budget/
  best-val/freeze); pre-repair checkpoints never used (carrier moved ch0→ch5).
- Specialists on repaired pure-family subsets (120 train / 40 val); joints on
  repaired F+R+C; router on all-family repaired mixed (num_experts=7 per the
  Phase 12 4→5 portfolio-sizing precedent; architecture/loss/temperature unchanged).
- Expert epochs 40 (phases 13–18 protocol), router epochs 30 (phase 8b default).
- Seeds 11/23/37; staged + cached; fingerprints + zero-overlap verified.

## Cross-evaluation matrix (repaired, mean over seeds)

| Expert | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp | 99.7 | 52.8 | 49.2 | 76.7 | 45.6 | 88.3 | 57.2 |
| graph | 50.0 | 62.8 | 49.2 | 46.9 | 51.9 | 50.0 | 52.2 |
| attention | 50.0 | 54.4 | 55.0 | 61.1 | 48.9 | 48.9 | 39.2 |
| attention_v2 | 41.7 | 46.7 | 99.7 | 63.9 | 55.0 | 55.8 | 63.9 |
| joint | 100.0 | 65.6 | 99.7 | 76.1 | 53.3 | 73.1 | 76.1 |
| joint_co | 100.0 | 66.1 | 99.7 | 76.1 | 53.1 | 76.1 | 76.4 |
| depth3 | 100.0 | 67.5 | 100.0 | 71.7 | 52.8 | 72.8 | 75.8 |

## Differential repair effects (task repair, NOT architecture gains)

- FC 88.3% (mlp) vs historical ~50%: F carrier rescued — exactly as erasure theory predicts.
- FR ≈ same (76.7%); FRC ≈ same (76.4%); R pure 62.8–67.5% (below historical 89% —
  attributed to 120 vs 360 training samples + noisy 40-sample val selection, NOT to
  the repair: the statistic proves 98% observability; flagged for follow-up).
- RC 55.0% best (v2) vs historical 53.9%: UNCHANGED despite proven observability
  (rule baseline 97.8% on this same data). Collapse persists identically
  (agree 98.0% / disagree 1.3%).

## Composition, routing, ceiling

- Fixed composition (oracle k=2/3, 6 frozen chains): RC gain +0.0pp, FRC −0.8pp —
  even best-combo logit averaging cannot fix RC (averaging C-predictors stays C).
- Learned routing beats best single expert +2.5pp mixed, 3/3 seeds (H6 SUPPORTED),
  but loses to the per-family ceiling −1.3pp (H8 NOT SUPPORTED).
- Compute-aware sweep behaves sanely (accuracy falls as λ rises); seed-stable.
- New empirical ceiling (repaired, evaluated portfolio): recorded in
  `empirical_ceiling.csv`; historical 66.4% cited for reference only.

## Component semantics on repaired RC (20K, the fair test at last)

- Agreement audit + input-level swaps: collapse persists with R observable.
- R-swap/C-swap change-rates, probes, and destruction in `component_semantics.csv`,
  `representation_probe.csv`, `causal_controls.csv`.

## Hypotheses (programmatic)

- H1 NOT SUPPORTED (worst pure R 67.5% — budget-attributed, see above).
- H2 SUPPORTED (best-per-family mixed 74.1%).
- H3 SUPPORTED (RC rule-vs-single margin +46.1pp: RC genuinely requires both).
- H4 NOT SUPPORTED (FRC margin −10.8pp: best single-component rule beats the
  constructed rule — probed-component noise cascades in 3-way majority).
- H5 NOT SUPPORTED. H6 SUPPORTED. H7 SUPPORTED (collapse persists).
- H8 NOT SUPPORTED (portfolio −1.3pp vs ceiling).

## Verdict (programmatic): CASE B

Composition failure persists despite valid observability — now a legitimate
research question (Outcome 2), no longer confounded by erasure. NO INTERVENTION
(20O): Phase 20 establishes the post-repair baseline.

## Failure flags

Pure-R budget weakness, mixed RC weakness, composition/routing-vs-ceiling gaps,
persistent collapse. Seed-stable throughout.

## Decision for Phase 21 (evidence-driven)

1. The RC decision on observable inputs is the precise open problem: models
   reproduce the C-side collapse (agree ~98 / disagree ~1) where a statistic
   rule reaches ~98%. Ask why the learned decision does not discover the
   R-side mapping — with R-branch capacity, not more of it, as the suspect
   (R pure itself is weak at 62–68%).
2. Routing helps (+2.5pp) but cannot beat the ceiling: route-quality diagnosis
   before expert changes.
3. Do NOT relitigate the benchmark: 8/8 gates + smoke pass on every run.
