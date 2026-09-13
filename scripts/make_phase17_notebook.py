"""Generate notebooks/25_heterogeneous_information_composition.ipynb (no hardcoded results)."""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


P = "../results/metrics/phase17_composition/summary.json"
C = "../results/metrics/phase17_composition/"

cells = [
    md("# Phase 25 — Heterogeneous Information Composition (Phase 17)\n"
       "## NeuroForge Experimental Research\n"
       "\n"
       "Phase 15: depth-3 improves isolated R. Phase 16: relational information is genuinely "
       "created; embedded computation NOT inferior; suppression/starvation NOT supported. "
       "Question: WHERE does information become incompatible when heterogeneous branches must be "
       "combined? Target: the composition interface. Distinguish exists / decodable / "
       "task-usable / composable — these are not interchangeable claims."),

    md("## 1. Phase 16 baseline"),
    code("import json\n"
         f"p17 = json.load(open('{P}', encoding='utf-8'))\n"
         "for k in ('baseline_perf_mean','depth3_perf_mean'):\n"
         "    d = p17[k]\n"
         "    print(f\"{k}: \" + '  '.join(f\"{f}={d.get(f,0)*100:.1f}%\" for f in ('F','R','C','FR','RC','FC','FRC')))\n"
         "print('Phase 16: relational info created (probe 85%%, drop 15.3pp) but RC/FRC unmoved.')"),

    md("## 2. Research question"),
    code("print('Where does information become incompatible when heterogeneous branches combine?')\n"
         "print('Candidates: interference, destructive addition, scale mismatch, non-separability,')\n"
         "print('single-state capacity, or task/composition limitation. No new fusion before the gate.')"),

    md("## 3. Representation preservation (17B: stages x components)"),
    code("pres = p17['aggregates']['preservation']['matrix_mean']\n"
         "for stage, probes in pres.items():\n"
         "    print(f\"{stage:>13}: \" + '  '.join(f\"{t}={v*100:.1f}%\" for t,v in probes.items()))\n"
         "print()\n"
         "print('Key Q: does independently-available branch info stay SIMULTANEOUSLY decodable after fusion?')"),

    md("## 4. Branch probe matrix (17C: Case 1-4 localization)"),
    code("for t in ('F_signal','R_signal','C_signal'):\n"
         "    pre = p17['aggregates']['preservation'][f'branch_{t}']\n"
         "    post = p17['aggregates']['preservation'][f'fused_{t}']\n"
         "    print(f\"{t}: branch {pre*100:.1f}% -> fused {post*100:.1f}% (drop {(pre-post)*100:+.1f}pp)\")\n"
         "print('Cases: 1=all decodable 2=relational lost 3=present but unusable 4=entangled')"),

    md("## 5. Scale diagnostics (17F: norms and domination ratios)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}scale_diagnostics.csv')):\n"
         f"    if row['seed'] == str(p17['per_seed_results'][0]['seed']):\n"
         "        print({k: (round(float(v),4) if (k != 'branch' and v not in ('', None)) else v) for k,v in row.items() if k != 'seed'})\n"
         "print('Purpose: does one branch numerically dominate the shared residual? (magnitude != importance)')"),

    md("## 6. Branch interaction (17G: interaction_gain = pair - best single)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}branch_interactions.csv')):\n"
         f"    if row['seed'] == str(p17['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['pair']:>4} {row['family']}: pair={float(row['pair_acc'])*100:.1f}% gain={float(row['interaction_gain'])*100:+.1f}pp\")\n"
         "print('Negative interaction alone is not proof of interference; combine with probes.')"),

    md("## 7. Composition order (17H: frozen chains; readout confound controlled)"),
    code("o = p17['aggregates']['order']\n"
         "print('raw means R:', {k: f'{v*100:.1f}%' for k,v in o['means_R'].items()})\n"
         "print('SAME-final-expert orientations (readout held fixed):')\n"
         "for name, d in o['orientations'].items():\n"
         "    print(f\"  {name}: {d['diff_mean']*100:+.1f}pp, positive {d['n_pos']}/{d['n_seeds']} seeds\")\n"
         "print('Raw spread tracks final-expert identity; controlled effects are seed-inconsistent.')"),

    md("## 8. Oracle representation (17E: full branch info vs fused state, one protocol)"),
    code("suff = p17['aggregates']['sufficiency']\n"
         "print(f\"oracle R/RC/FRC: {suff['oracle_R']*100:.1f}/{suff['oracle_RC']*100:.1f}/{suff['oracle_FRC']*100:.1f}%\")\n"
         "print(f\"gains over fused head: RC {suff['oracle_RC_gain']*100:+.1f}pp, FRC {suff['oracle_FRC_gain']*100:+.1f}pp\")\n"
         "print(f\"causal drops: oracle {suff['oracle_drop_R']*100:+.1f}pp vs fused {suff['fused_drop_R']*100:+.1f}pp\")"),

    md("## 9. Joint decodability (17J: R-only vs R+C vs F+R+C on RC/FRC)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}joint_decodability.csv')):\n"
         f"    if row['seed'] == str(p17['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['info']:>6}: R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")\n"
         "print('R good + R+C bad = interference. R+C improves = coexistence. None improve = deeper limit.')"),

    md("## 10. Frozen diagnostic composition (17K: diagnostic head vs production head)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}frozen_oracle.csv')):\n"
         f"    if row['seed'] == str(p17['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['head']:>11}: R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")"),

    md("## 11. RC/FRC analysis (17M primary criteria + agreement splits)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}component_agreement.csv')):\n"
         f"    if row['seed'] == str(p17['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['family']} {row['split']}: acc={float(row['acc'])*100:.1f}% (n={row['n']})\")\n"
         "print('RC-disagree ~0%% means: knows components, cannot combine them.')"),

    md("## 12. Causal controls (destruction alongside absolute accuracy)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}causal_controls.csv')):\n"
         f"    if row['seed'] == str(p17['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['condition']:>9} {row['variant']:>20}: R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")"),

    md("## 13. Efficiency (params/FLOPs/latency/activation; FLOPs != latency)"),
    code(f"import csv\n"
         f"for name in ('compute.csv','latency.csv'):\n"
         "    print(f'--- {name} ---')\n"
         f"    for row in csv.DictReader(open('{C}' + name)):\n"
         f"        if row['seed'] == str(p17['per_seed_results'][0]['seed']):\n"
         "            print(' ', {k: v for k,v in row.items() if k != 'seed'})"),

    md("## 14. H1-H8 (re-derived programmatically)"),
    code("from neuroforge.evaluation.phase17_composition_diagnostics import build_phase17_hypotheses\n"
         "hyps = build_phase17_hypotheses(p17['aggregates'])\n"
         "for h in sorted(hyps):\n"
         "    print(f\"{h}: {hyps[h]['status']}\")\n"
         "    print(f\"    {hyps[h]['evidence']}\")\n"
         "assert all(hyps[h]['status'] == p17['hypotheses'][h]['status'] for h in hyps)\n"
         "print('stored verdicts match fresh derivation: OK')"),

    md("## 15. Final CASE (programmatic)"),
    code("from neuroforge.evaluation.phase17_composition_diagnostics import select_phase17_case\n"
         "case, label = select_phase17_case(hyps, p17['aggregates'])\n"
         "print(f'Programmatic verdict: {case} — {label}')\n"
         "assert case == p17['verdict_case']\n"
         "print(f\"Minimal intervention: {p17['minimal_intervention']['intervention']} ({p17['minimal_intervention']['outcome']})\")"),

    md("## 16. Evidence-backed next step (programmatic recommendation)"),
    code("from neuroforge.evaluation.phase17_composition_diagnostics import recommendation_for_case\n"
         "print('Recommendation:', recommendation_for_case(p17['verdict_case']))\n"
         "print()\n"
         "print('Loaded (not typed):')\n"
         "print(f\"  preservation worst drop: {min(p17['aggregates']['preservation'][t] for t in ('F_signal_drop_pre_to_fused','R_signal_drop_pre_to_fused','C_signal_drop_pre_to_fused'))*100:+.1f}pp\")\n"
         "print(f\"  oracle RC/FRC gains: {p17['aggregates']['sufficiency']['oracle_RC_gain']*100:+.1f}/{p17['aggregates']['sufficiency']['oracle_FRC_gain']*100:+.1f}pp\")\n"
         "print(f\"  domination ratio: {p17['aggregates']['domination']['max_min_branch_mean']:.2f}x\")"),
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

out = Path("notebooks/25_heterogeneous_information_composition.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
