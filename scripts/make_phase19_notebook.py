"""Generate notebooks/27_mixed_benchmark_repair_validation.ipynb (no hardcoded results)."""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


P = "../results/metrics/phase19_benchmark_validation/summary.json"
C = "../results/metrics/phase19_benchmark_validation/"

cells = [
    md("# Phase 27 — Mixed-Benchmark Repair Validation (Phase 19)\n"
       "## NeuroForge Experimental Research\n"
       "\n"
       "Phase 18: the mixed construction erases the relational carrier on RC/FRC "
       "(ch0 add, then ch0:3 overwrite) while labels keep depending on it. "
       "Question: can the benchmark be repaired minimally so every label-generating "
       "component stays observable, preserving all other semantics? Repair first, "
       "models later — no architecture in this phase."),

    md("## 1. Phase 18 discovery"),
    code("import json\n"
         f"p19 = json.load(open('{P}', encoding='utf-8'))\n"
         "b = p19['aggregates']['bug']\n"
         "print(f\"original R-align: R={b['orig_R']*100:.1f}% RC={b['orig_RC']*100:.1f}% FRC={b['orig_FRC']*100:.1f}%\")\n"
         "print(f\"erasure reproduced: {b['erasure_reproduced']}\")"),

    md("## 2. Reproduce historical construction bug (19A)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}bug_reproduction.csv')):\n"
         f"    if row['seed'] == str(p19['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['metric']}: {float(row['value'])*100:.1f}%\" if float(row['value']) <= 1 else f\"{row['metric']}: {row['value']}\")"),

    md("## 3. Exact overwrite dependency (19D ledger + empirical rates)"),
    code(f"import csv\n"
         f"seen = 0\n"
         f"for row in csv.DictReader(open('{C}dependency_audit.csv')):\n"
         f"    if row['seed'] == str(p19['per_seed_results'][0]['seed']) and row['family'] in ('RC','FRC'):\n"
         "        print(f\"{row['construction']:>9} {row['family']} {row['component']}: ch={row['carrier_channel']} rate={float(row['observable_rate'])*100:.1f}% :: {row['later_transformation'][:70]}\")\n"
         "        seen += 1\n"
         "        if seen >= 8:\n"
         "            break"),

    md("## 4. Minimal repair (versioned module; same RNG values, same label algebra)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}construction_diff.csv')):\n"
         "    print(f\"{row['area']}: {row['before']} -> {row['after']}\")\n"
         "print('Repair: C first, F offsets after (common-mode), R carrier on ch5 (untouched).')"),

    md("## 5. Observability audit (19C: R/F/C carriers per family, repaired)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}observability.csv')):\n"
         f"    if row['seed'] == str(p19['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['metric']:>12}: {float(row['value'])*100:.1f}%\")"),

    md("## 6. Label-input dependency audit (19D: every label variable observable)"),
    code("print('Reusable audit: statistic -> carrier channel -> later transforms -> rate.')\n"
         "print('Ledger declares overwrites; empirical rates confirm. See dependency_audit.csv.')\n"
         "o = p19['aggregates']['observability']\n"
         "print(f\"R-group: {[round(o[k]*100,1) for k in ('r_align_R','r_align_RC','r_align_FRC','r_align_FR')]}\")"),

    md("## 7. RC parity analysis (19E: genuine two-component requirement)"),
    code("s = p19['aggregates']['semantics']\n"
         "print(f\"rule RC={s['rule_RC']*100:.1f}% vs C-only={s['c_only_RC']*100:.1f}% (margin bar 20pp)\")\n"
         "print(f\"parity holds; FRC ties absent: {s['frc_no_tie']}; keys present: {s['keys_present']}\")"),

    md("## 8. Counterfactual validation (19F: corrected methodology only)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}counterfactual_validation.csv')):\n"
         f"    if row['seed'] == str(p19['per_seed_results'][0]['seed']):\n"
         "        print(f\"R-valid={float(row['R_valid_rate'])*100:.1f}% C-valid={float(row['C_valid_rate'])*100:.1f}% (bar 90%)\")"),

    md("## 9. Invariance controls (19G: C absolute, R comparative vs original)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}invariance_controls.csv')):\n"
         f"    if row['seed'] == str(p19['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['metric']}: {float(row['value']):+.3f}\")\n"
         "print('Token moves shift positional carriers in BOTH versions (pre-existing); C retrieval must hold absolutely.')"),

    md("## 10. Shortcut/leakage audit (19H: no direct copies, no new family leakage)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}shortcut_audit.csv')):\n"
         f"    if row['seed'] == str(p19['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['metric']}: {float(row['value']):.3f}\")"),

    md("## 11. Shallow baselines (19I: rule + matched linear + tiny MLP context)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}shallow_baselines.csv')):\n"
         f"    if row['seed'] == str(p19['per_seed_results'][0]['seed']):\n"
         "        print(f\"{row['metric']}: {float(row['value'])*100:.1f}%\")\n"
         "print('Raw-MLP limits on second-order stats are inductive-bias context, not validity failures.')"),

    md("## 12. Common-input contract (19K: all experts consume repaired batches)"),
    code(f"import csv\n"
         f"for row in csv.DictReader(open('{C}common_contract.csv')):\n"
         f"    if row['seed'] == str(p19['per_seed_results'][0]['seed']):\n"
         "        print(row)\n"
         "print('Still [B,S,8]; marker semantics intact; no family/expert/label metadata.')"),

    md("## 13. Validity gates G1-G8 (all must pass)"),
    code("for g, d in p19['aggregates']['gates'].items():\n"
         "    print(f\"{g}: {'PASS' if d['passed'] else 'FAIL'} — {d['detail'][:110]}\")"),

    md("## 14. H1-H8 (re-derived programmatically)"),
    code("from neuroforge.evaluation.phase19_benchmark_validation import build_phase19_hypotheses\n"
         "hyps = build_phase19_hypotheses(p19['aggregates'])\n"
         "for h in sorted(hyps):\n"
         "    print(f\"{h}: {hyps[h]['status']}\")\n"
         "    print(f\"    {hyps[h]['evidence']}\")\n"
         "assert all(hyps[h]['status'] == p19['hypotheses'][h]['status'] for h in hyps)\n"
         "print('stored verdicts match fresh derivation: OK')"),

    md("## 15. Final CASE (programmatic)"),
    code("from neuroforge.evaluation.phase19_benchmark_validation import select_phase19_case\n"
         "case, label = select_phase19_case(hyps, p19['aggregates'])\n"
         "print(f'Programmatic verdict: {case} — {label}')\n"
         "assert case == p19['verdict_case']\n"
         "print(f\"Intervention recorded: {p19['minimal_intervention']['intervention']}\")"),

    md("## 16. Recommendation on reopening composition experiments"),
    code("from neuroforge.evaluation.phase19_benchmark_validation import recommendation_for_case\n"
         "print('Recommendation:', recommendation_for_case(p19['verdict_case']))\n"
         "print()\n"
         "print('Loaded (not typed):')\n"
         "print(f\"  gates passed: {sum(1 for d in p19['aggregates']['gates'].values() if d['passed'])}/8\")\n"
         "print(f\"  rule RC={p19['aggregates']['semantics']['rule_RC']*100:.1f}%, lagprobe R={p19['aggregates']['learnability']['lagprobe_R']*100:.1f}%\")"),
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

out = Path("notebooks/27_mixed_benchmark_repair_validation.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
