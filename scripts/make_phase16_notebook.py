"""Generate notebooks/24_relational_interface_and_coadaptation.ipynb (no hardcoded results)."""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


P = "../results/metrics/phase16_relational_interface/summary.json"
C = "../results/metrics/phase16_relational_interface/"

cells = [
    md("# Phase 24 — Relational Interface and Co-adaptation (Phase 16)\n"
       "## NeuroForge Experimental Research\n"
       "\n"
       "Central observation (Phase 15): the dedicated Graph specialist exploits relational "
       "structure strongly (≈84–93% R, ≈32–47pp destruction drops) while the JointCo embedded "
       "path does not (≈5pp drops). Question: WHY? Localize among input/interface, "
       "co-adaptation, post-branch loss, training signal, or effective computation. "
       "Localization only — no architecture is earned until the discrepancy is understood."),

    md("## 1. Environment verification"),
    code("import platform, torch, neuroforge\n"
         "print(f'Python: {platform.python_version()}')\n"
         "print(f'PyTorch: {torch.__version__}')\n"
         "print(f'NeuroForge: {neuroforge.__file__}')"),

    md("## 2. Exact baselines (16A — executed, never historical numbers)"),
    code(f"import json\n"
         f"p16 = json.load(open('{P}', encoding='utf-8'))\n"
         "for k in ('graph_perf_mean','baseline_perf_mean','depth3_perf_mean','rel_focused_perf_mean','rel_first_perf_mean'):\n"
         "    d = p16[k]\n"
         "    print(f\"{k:>22}: \" + '  '.join(f\"{f}={d.get(f,0)*100:.1f}%\" for f in ('F','R','C','FR','RC','FC','FRC')))\n"
         "print('reproduction vs Phase 15:', p16['baseline_check_vs_phase15'])"),

    md("## 3. Relational input equivalence (16B — same raw batch, different encoders)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}input_equivalence.csv')):\n"
         f"    if row['seed'] == str(p16['per_seed_results'][0]['seed']):\n"
         "        print(row)\n"
         "print('Question answered: are we comparing the same relational problem at the input?')"),

    md("## 4. Graph-on-Joint-input (16C — THE critical experiment: same input, different computation)"),
    code("c = p16['aggregates']['computation']\n"
         "print(f\"embedded branch + head R: {c['embedded_R']*100:.1f}%\")\n"
         "print(f\"graph-on-joint-input + head R: {c['graph_on_joint_R']*100:.1f}%\")\n"
         "print(f\"same-input gap: {c['graph_on_joint_minus_embedded_R']*100:+.1f}pp\")\n"
         "print(f\"fresh graph diagnostic R: {p16['aggregates']['graph_diagnostic']['diagnostic_R']*100:.1f}%\")\n"
         "print('If graph-on-joint >> embedded: computation suspect. If similar: input suspect.')"),

    md("## 5. Joint-input Graph probe (16D — can ANY Graph computation exploit the joint input?)"),
    code("g = p16['aggregates']['graph_diagnostic']\n"
         "print(f\"fresh graph diagnostic R: {g['diagnostic_R']*100:.1f}% (probe {g['diagnostic_probe_R']*100:.1f}%)\")\n"
         "print(f\"native Graph R (reference): {p16['graph_perf_mean']['R']*100:.1f}%\")\n"
         "print(f\"native-minus-diagnostic: {g['native_minus_diagnostic_R']*100:+.1f}pp\")"),

    md("## 6. Pre/post relational analysis (16E — probe + usability + causal dependence)"),
    code("st = p16['aggregates']['stages']\n"
         "print(f\"pre-rel probe {st['pre_probe_R']*100:.1f}% / head {st['pre_R']*100:.1f}% / drop {st['pre_drop_R']*100:+.1f}pp\")\n"
         "print(f\"post-rel probe {st['post_probe_R']*100:.1f}% / head {st['post_R']*100:.1f}% / drop {st['post_drop_R']*100:+.1f}pp\")\n"
         "print(f\"post-fusion probe {st['post_fusion_probe_R']*100:.1f}% / head {st['post_fusion_R']*100:.1f}%\")\n"
         "print(f\"native-input probe {st['native_input_probe_R']*100:.1f}% / final R {st['final_R']*100:.1f}%\")"),

    md("## 7. Branch-freeze co-adaptation (16F — is joint optimization suppressive?)"),
    code("co = p16['aggregates']['coadaptation']\n"
         "print(f\"joint (depth3) R: {co['joint_R']*100:.1f}%\")\n"
         "print(f\"rel-focused R: {co['relfocused_R']*100:.1f}% ({co['relfocused_minus_joint_R']*100:+.1f}pp)\")\n"
         "print(f\"rel-first R: {co['relfirst_R']*100:.1f}% ({co['relfirst_minus_joint_R']*100:+.1f}pp)\")"),

    md("## 8. Gradient/signal diagnosis (16G — diagnostic instrumentation only)"),
    code("gr = p16['aggregates']['gradients']\n"
         "print('status:', gr.get('status'))\n"
         "print('group means:', {k: round(v,4) for k,v in gr.get('group_means', {}).items()})\n"
         "print('observed:', gr.get('observed'))"),

    md("## 9. Branch ablation on the depth-3 candidate (16H)"),
    code(f"import csv\n"
         f"best = p16['best_cond']\n"
         f"for row in csv.DictReader(open('{C}branch_ablation.csv')):\n"
         f"    if row['seed'] == str(p16['per_seed_results'][0]['seed']) and row['condition'] == ('depth3' if best != 'depth3' else 'depth3'):\n"
         "        print(f\"{row['combination']:>6}: R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")\n"
         "print('Question: does relational computation degrade specifically when other branches are present?')"),

    md("## 10. Representation compatibility (16I — raw vs joint-pre vs native input)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}graph_on_joint_input.csv')):\n"
         f"    if row['seed'] == str(p16['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['condition']:>26}: R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")"),

    md("## 11. Destruction as causal metric (16J) + composition (16K)"),
    code("se = p16['aggregates']['sensitivity']\n"
         "print(f\"baseline drop {se['baseline_drop_R']*100:+.1f}pp vs {se['candidate']} drop {se['candidate_drop_R']*100:+.1f}pp\")\n"
         "cp = p16['aggregates']['compositional']\n"
         "print(f\"strongest ({cp['condition']}): R {cp['R_gain']*100:+.1f}pp, RC {cp['RC_gain']*100:+.1f}pp, FRC {cp['FRC_gain']*100:+.1f}pp\")\n"
         "print('High R + high drop differs materially from similar R + low drop.')"),

    md("## 12. Compute, latency, seeds"),
    code("import csv\n"
         "for _name in ('compute.csv','latency.csv','seed_results.csv'):\n"
         "    print(f'--- {_name} ---')\n"
         f"    for i, row in enumerate(csv.DictReader(open('{C}' + _name))):\n"
         "        if i < 6:\n"
         "            print(' ', dict(row))\n"
         "print('stability:', {k: {c: f'{v*100:.1f}pp' for c,v in d.items()} for k,d in p16['aggregates']['seed_stability'].items()})"),

    md("## 13. H1–H8 (re-derived programmatically)"),
    code("from neuroforge.evaluation.phase16_metrics import build_phase16_hypotheses\n"
         "hyps = build_phase16_hypotheses(p16['aggregates'])\n"
         "for h in sorted(hyps):\n"
         "    print(f\"{h}: {hyps[h]['status']}\")\n"
         "    print(f\"    {hyps[h]['evidence']}\")\n"
         "assert all(hyps[h]['status'] == p16['hypotheses'][h]['status'] for h in hyps)\n"
         "print('stored verdicts match fresh derivation: OK')"),

    md("## 14. Final CASE + evidence-backed recommendation (programmatic)"),
    code("from neuroforge.evaluation.phase16_metrics import select_phase16_case, recommendation_for_case\n"
         "case, label = select_phase16_case(hyps, p16['aggregates']['stages'])\n"
         "print(f'Programmatic verdict: {case} — {label}')\n"
         "assert case == p16['verdict_case']\n"
         "print(f'Minimal intervention: {p16[\"minimal_intervention\"][\"intervention\"]} ({p16[\"minimal_intervention\"][\"outcome\"]})')\n"
         "print(f'Recommendation: {recommendation_for_case(case)}')\n"
         "print()\n"
         "print('Evidence (loaded, not typed):')\n"
         "print(f\"  same-input gap {p16['aggregates']['computation']['graph_on_joint_minus_embedded_R']*100:+.1f}pp; \"+\n"
         "      f\"rel-focused {p16['aggregates']['coadaptation']['relfocused_minus_joint_R']*100:+.1f}pp; \"+\n"
         "      f\"pre/post probes {p16['aggregates']['stages']['pre_probe_R']*100:.1f}/{p16['aggregates']['stages']['post_probe_R']*100:.1f}%; \"+\n"
         "      f\"grad status {p16['aggregates']['gradients']['status']}\")"),
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

out = Path("notebooks/24_relational_interface_and_coadaptation.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
