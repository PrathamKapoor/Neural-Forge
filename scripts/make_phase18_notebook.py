"""Generate notebooks/26_component_composition_diagnosis.ipynb (no hardcoded results)."""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


P = "../results/metrics/phase18_component_composition/summary.json"
C = "../results/metrics/phase18_component_composition/"

cells = [
    md("# Phase 26 — Component Composition Diagnosis (Phase 18)\n"
       "## NeuroForge Experimental Research\n"
       "\n"
       "Phase 17: branch info survives fusion; RC-agree ~100% vs RC-disagree ~0-2%. "
       "Question: WHY does the predictor select one available component when the task "
       "requires combining components? Decision-semantic diagnosis across: label semantics, "
       "component vs joint prediction, representation availability, head behavior, loss "
       "incentives, calibration, RC-vs-FRC structure. RC uses R+C semantics per construction "
       "(F is a negative control)."),

    md("## 1. Phase 17 baseline"),
    code("import json\n"
         f"p18 = json.load(open('{P}', encoding='utf-8'))\n"
         "for k in ('baseline_perf_mean','depth3_perf_mean'):\n"
         "    d = p18[k]\n"
         "    print(f\"{k}: \" + '  '.join(f\"{f}={d.get(f,0)*100:.1f}%\" for f in ('F','R','C','FR','RC','FC','FRC')))\n"
         "print('Phase 17: info survives fusion; RC-disagree collapses.')"),

    md("## 2. Research question"),
    code("print('Why select one available component when the task requires combining?')\n"
         "print('Distinguish: semantics / component-vs-joint / availability / head / loss / calibration / RC-vs-FRC.')"),

    md("## 3. Component target audit (18B: diagnostic metadata, official labels untouched)"),
    code(f"import csv\n"
         f"n = 0\n"
         f"for row in csv.DictReader(open('{C}component_targets.csv')):\n"
         "    n += 1\n"
         "    if n <= 3:\n"
         "        print(row)\n"
         f"print(f'... total RC/FRC metadata rows: {{n}}')\n"
         "print('RC components are R+C per construction (F is a negative control here).')"),

    md("## 4. Component prediction matrix (18C: rep x component, incl. F negative control)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}component_probe_matrix.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['representation']:>6}: \" + '  '.join(f\"{k}={float(v)*100:.1f}%\" for k,v in row.items() if k not in ('seed','representation')))\n"
         "print('F-on-RC must read ~chance (no F injected): validates probes are honest.')"),

    md("## 5. Joint decodability (18D: 4-class (sr,sc) target on RC)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}joint_decodability.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['representation']:>6}: joint={float(row['joint_acc'])*100:.1f}% R-marg={float(row['R_marginal'])*100:.1f}% C-marg={float(row['C_marginal'])*100:.1f}%\")"),

    md("## 6. RC agreement/disagreement (18E, central: production logits/margins/norms)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}agreement_disagreement.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['split']:>9}: acc={float(row['acc'])*100:.1f}% margin={float(row['margin']):.3f} conf={float(row['confidence']):.3f} ent={float(row['entropy']):.3f}\")\n"
         "print('Q: does failure concentrate where heterogeneous evidence must be resolved?')"),

    md("## 7. Logits and margins (18F: which component is favored; correlation only)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}logit_margin.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print(f\"match C={float(row['matches_C_disagree'])*100:.1f}% match R={float(row['matches_R_disagree'])*100:.1f}% | \"+\n"
         "              f\"rel/dis={float(row['rel_norm_disagree']):.3f} ctx/dis={float(row['ctx_norm_disagree']):.3f}\")\n"
         "print('Do NOT infer causality from correlation (§25).')"),

    md("## 8. Counterfactual evidence (18G: change-rate is the valid responsiveness test)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}counterfactuals.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['swap']}: flip={float(row['flip_rate'])*100:.1f}%\")\n"
         "print('Implied-match cannot distinguish (a C-predictor matches both); change-rate can.')\n"
         "cf = p18['aggregates']['counterfactual']\n"
         "print(f\"change-rates: R-swap {cf['R_swap_response']*100:.1f}% vs C-swap {cf['C_swap_response']*100:.1f}% (control {cf['control_same_response']*100:.1f}%)\")"),

    md("## 9. Diagnostic heads (18H: linear vs nonlinear on frozen R+C)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}diagnostic_heads.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print({k: v for k,v in row.items() if k != 'seed'})\n"
         "print('Linear ok -> production limitation. Nonlinear-only -> nonlinear combination. Both fail -> deeper.')"),

    md("## 10. Objective diagnosis (18I: feasible rules, not label-oracles)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}objective_diagnosis.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print(f\"model tr/te={float(row['model_train_RC'])*100:.1f}/{float(row['model_test_RC'])*100:.1f}% | \"+\n"
         "              f\"R-rule(tr/te)={float(row['rule_R_train_RC'])*100:.1f}/{float(row['rule_R_test_RC'])*100:.1f}% | \"+\n"
         "              f\"C-rule(tr/te)={float(row['rule_C_train_RC'])*100:.1f}/{float(row['rule_C_test_RC'])*100:.1f}%\")\n"
         "print('R-rule is label-oracle (sr erased); judge shortcut on the feasible C-rule.')"),

    md("## 11. RC/FRC semantic contrast (18J: erasure proof + parity algebra)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}rc_frc_semantics.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print(f\"parity={row['parity_fact']} C-match={float(row['production_matches_C_rule'])*100:.1f}% | \"+\n"
         "              f\"align R/RC/FRC={float(row['align_R'])*100:.1f}/{float(row['align_RC'])*100:.1f}/{float(row['align_FRC'])*100:.1f}%\")\n"
         "print('align: autocorr-stat vs sr. High on R validates it; chance on RC/FRC proves erasure.')\n"
         "print('Construction: cand added to ch0, then channels 0:3 overwritten (verified by reading).')"),

    md("## 12. Shuffled controls (18K: no accidental correlation)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}shuffled_controls.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print(f\"label-shuffled={float(row['label_shuffled_RC'])*100:.1f}% R-permuted={float(row['R_permuted_RC'])*100:.1f}% (must be ~chance)\")"),

    md("## 13. Intervention gate (18L: two converging observations required)"),
    code("print('gate:', p18['gate'])\n"
         "print('Default is NO INTERVENTION.')"),

    md("## 14. Optional minimal intervention (18M: only if gate passes)"),
    code("print('intervention record:', p18['aggregates']['intervention'])\n"
         "print('minimal intervention:', p18['minimal_intervention'])"),

    md("## 15. RC/FRC evaluation (18N primary criteria)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}rc_frc_results.csv')):\n"
         f"    if row['seed'] == str(p18['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['condition']:>9}: R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")"),

    md("## 16. Empirical ceiling (18O: never redefined)"),
    code("c = p18['aggregates']['ceiling']\n"
         "print(f\"{c['old_ceiling_mixed_mean']*100:.1f}% -> {c['new_ceiling_mixed_mean']*100:.1f}% ({c['new_minus_old_mixed']*100:+.1f}pp)\")"),

    md("## 17. H1-H8 (re-derived programmatically)"),
    code("from neuroforge.evaluation.phase18_component_composition import build_phase18_hypotheses\n"
         "hyps = build_phase18_hypotheses(p18['aggregates'])\n"
         "for h in sorted(hyps):\n"
         "    print(f\"{h}: {hyps[h]['status']}\")\n"
         "    print(f\"    {hyps[h]['evidence']}\")\n"
         "assert all(hyps[h]['status'] == p18['hypotheses'][h]['status'] for h in hyps)\n"
         "print('stored verdicts match fresh derivation: OK')"),

    md("## 18. Final CASE (programmatic)"),
    code("from neuroforge.evaluation.phase18_component_composition import select_phase18_case\n"
         "case, label = select_phase18_case(hyps, p18['aggregates'])\n"
         "print(f'Programmatic verdict: {case} — {label}')\n"
         "assert case == p18['verdict_case']\n"
         "print(f\"Minimal intervention: {p18['minimal_intervention']['intervention']} ({p18['minimal_intervention']['outcome']})\")"),

    md("## 19. Next-step recommendation (programmatic)"),
    code("from neuroforge.evaluation.phase18_component_composition import recommendation_for_case\n"
         "print('Recommendation:', recommendation_for_case(p18['verdict_case']))\n"
         "print()\n"
         "print('Loaded (not typed):')\n"
         "s = p18['aggregates']['semantics']\n"
         "print(f\"  erasure={s['erasure_demonstrated']}, align R/RC={[round(s[k]*100,1) for k in ('align_R','align_RC')]}\")\n"
         "print(f\"  disagree={p18['aggregates']['agreement']['disagree_acc']*100:.1f}%, C-favor={p18['aggregates']['favor']['disagree_pred_matches_C']*100:.1f}%\")\n"
         "print(f\"  R-swap response={p18['aggregates']['counterfactual']['R_swap_response']*100:.1f}%, C-swap={p18['aggregates']['counterfactual']['C_swap_response']*100:.1f}%\")"),
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

out = Path("notebooks/26_component_composition_diagnosis.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
