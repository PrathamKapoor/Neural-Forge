"""Generate notebooks/28_repaired_portfolio_re_evaluation.ipynb (no hardcoded results)."""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


P = "../results/metrics/phase20_repaired_portfolio/summary.json"
C = "../results/metrics/phase20_repaired_portfolio/"
EXPERTS = "('mlp','graph','attention','attention_v2','joint','joint_co','depth3')"

cells = [
    md("# Phase 28 — Repaired Portfolio Re-evaluation (Phase 20)\n"
       "## NeuroForge Experimental Research\n"
       "\n"
       "Phase 19 repaired and validated the mixed benchmark (phase19-repaired-v1, 8/8 gates). "
       "Phase 20 re-evaluates the EXISTING portfolio with NO architecture changes: fresh "
       "training on repaired data only (pre-repair checkpoints are invalid — carrier moved "
       "ch0→ch5). Question: with all components genuinely observable, does the portfolio "
       "demonstrate compositional capability?"),

    md("## 1. Phase 19 validity result"),
    code("import json\n"
         f"p20 = json.load(open('{P}', encoding='utf-8'))\n"
         "print('benchmark:', p20['manifest']['benchmark_version'])\n"
         "print('protocol:', p20['manifest']['protocol'][:150])"),

    md("## 2. Repaired benchmark smoke test (20A: guard against the old generator)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}benchmark_smoke.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['check']}: {row['value']}\")"),

    md("## 3. Fresh dataset generation (20B: fingerprints + zero overlap)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}dataset_manifest.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(row)"),

    md("## 4. Pure expert evaluation (20C: fresh training, repaired families)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}pure_experts.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['expert']:>12}: \" + '  '.join(f\"{f}={float(row[f])*100:.1f}%\" for f in ('F','R','C','FR','RC','FC','FRC')))"),

    md("## 5. Cross-evaluation matrix (20D: expert × family, mean±SD in CSV)"),
    code("cm = p20['cross_mean']\n"
         "for e in " + EXPERTS + ":\n"
         "    d = cm.get(e, {})\n"
         "    print(f\"{e:>12}: \" + '  '.join(f\"{f}={d.get(f,0)*100:.1f}%\" for f in ('F','R','C','FR','RC','FC','FRC')))\n"
         "print('Expected specialization pattern: check F-mlp, R-graph, C-v2.')"),

    md("## 6. New empirical ceiling (20E: repaired benchmark, evaluated portfolio)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}empirical_ceiling.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"seed {row['seed']}: \" + '  '.join(f\"{f}={float(row[f])*100:.1f}%\" for f in ('FR','RC','FC','FRC')))\n"
         "print('(Historical 66.4%% cited for reference only; never the new ceiling.)')"),

    md("## 7. Fixed composition (20F: oracle k=1..3 + frozen sequential chains)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}fixed_composition.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['policy']:>22}: \" + '  '.join(f\"{f}={float(row[f])*100:.1f}%\" for f in ('R','RC','FRC')))"),

    md("## 8. Oracle diagnostics (20G: component-availability vs optimal-path; mixed NOT DEFINED)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}oracle.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['policy']:>12}: \" + '  '.join(f\"{f}={float(row[f])*100:.1f}%\" for f in ('R','RC','FRC')))"),

    md("## 9. Random routing (seeded uniform baseline, no training)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}random_routing.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"mixed={float(row['mixed_mean'])*100:.1f}% R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")"),

    md("## 10. Learned routing (20H: machinery unchanged, 7-expert portfolio)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}learned_routing.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"mixed={float(row['mixed_mean'])*100:.1f}% R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")\n"
         "print('Router architecture/loss/temperature unchanged; num_experts=7 follows Phase 12 4→5 precedent.')"),

    md("## 11. Compute-aware routing (20J: existing λ values only)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}compute_aware.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"λ={row['lambda']:>6}: mixed={float(row['mixed_mean'])*100:.1f}% RC={float(row['RC'])*100:.1f}% lat={float(row['latency_us']):.1f}us\")"),

    md("## 12. Component semantics (20K: agreement + input-level swaps on repaired RC)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}component_semantics.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"agree={float(row['agree_acc'])*100:.1f}% disagree={float(row['disagree_acc'])*100:.1f}% \"+\n"
         "              f\"R-swapΔ={float(row['R_swap_response'])*100:.1f}% C-swapΔ={float(row['C_swap_response'])*100:.1f}%\")"),

    md("## 13. RC analysis (20L primary)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}rc_frc_analysis.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['condition']:>9}: R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")"),

    md("## 14. FRC analysis (20L primary)"),
    code("print('See rc_frc_analysis.csv FRC column above; oracle diagnostic FRC in fixed_composition.csv.')\n"
         "print('Q: does repaired observability move FRC beyond the C/F-majority baseline?')"),

    md("## 15. Representation probes (20M: information → joint → prediction, now valid)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}representation_probe.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['probe']:>14}: {float(row['value'])*100:.1f}%\")"),

    md("## 16. Relational causal controls (20N: graph/joint_co/best-composition)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}causal_controls.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['condition']:>10} {row['variant']:>20}: R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")"),

    md("## 17. Efficiency (params/FLOPs/latency/throughput)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}compute.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['expert']:>12}: params={row['params']} flops={float(row['flops']):.0f}\")\n"
         f"for row in csv.DictReader(open('{C}latency.csv')):\n"
         f"    if row['seed'] == str(p20['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['expert']:>12}: lat={float(row['latency_us']):.1f}us\")"),

    md("## 18. Historical-vs-repaired comparison (repair effects, NOT architecture gains)"),
    code("print('Original RC ~54% (C-rule on erased inputs) vs repaired RC (see §13).')\n"
         "print('Original FC ~50% (F erased) vs repaired FC (see §4 mlp row).')\n"
         "print('Deltas vs Phase 9-18 are task-repair effects unless otherwise controlled.')"),

    md("## 19. H1–H8 (re-derived programmatically)"),
    code("from neuroforge.evaluation.phase20_repaired_portfolio import build_phase20_hypotheses\n"
         "hyps = build_phase20_hypotheses(p20['aggregates'])\n"
         "for h in sorted(hyps):\n"
         "    print(f\"{h}: {hyps[h]['status']}\")\n"
         "    print(f\"    {hyps[h]['evidence']}\")\n"
         "assert all(hyps[h]['status'] == p20['hypotheses'][h]['status'] for h in hyps)\n"
         "print('stored verdicts match fresh derivation: OK')"),

    md("## 20. Final CASE (programmatic)"),
    code("from neuroforge.evaluation.phase20_repaired_portfolio import select_phase20_case\n"
         "case, label = select_phase20_case(hyps, p20['aggregates'])\n"
         "print(f'Programmatic verdict: {case} — {label}')\n"
         "assert case == p20['verdict_case']\n"
         "print(f\"Minimal intervention: {p20['minimal_intervention']['intervention']} ({p20['minimal_intervention']['outcome']})\")"),

    md("## 21. Next-step recommendation (programmatic)"),
    code("from neuroforge.evaluation.phase20_repaired_portfolio import recommendation_for_case\n"
         "print('Recommendation:', recommendation_for_case(p20['verdict_case']))\n"
         "print()\n"
         "print('Loaded (not typed):')\n"
         "print(f\"  best RC={p20['aggregates']['mixed']['best_RC']*100:.1f}%, best mixed={p20['aggregates']['mixed']['best_mixed_mean']*100:.1f}%\")\n"
         "print(f\"  routing gain={p20['aggregates']['routing']['learned_minus_best_single_mixed']*100:+.1f}pp, \"+\n"
         "      f\"ceiling delta={p20['aggregates']['ceiling']['portfolio_minus_single_mixed']*100:+.1f}pp\")"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/28_repaired_portfolio_re_evaluation.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
