"""Generate notebooks/23_relational_substep_diagnosis.ipynb (code-generated, no hardcoded results)."""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md("# Phase 23 — Relational Substep Diagnosis (Phase 15)\n"
       "## NeuroForge Experimental Research\n"
       "\n"
       "Research question: can a minimal, controlled increase or modification of relational "
       "computation convert the existing decodable-but-not-task-usable relational signal into "
       "stronger relational prediction? Distinguish: (1) message-passing depth, "
       "(2) neighbourhood aggregation, (3) relational update capacity, (4) fixed topology, "
       "(5) branch interaction. No assumption is made before experimentation."),
    md("## 1. Environment verification"),
    code("import platform, torch, neuroforge\n"
         "from pathlib import Path\n"
         "print(f'Python: {platform.python_version()}')\n"
         "print(f'PyTorch: {torch.__version__}')\n"
         "print(f'NeuroForge: {neuroforge.__file__}')"),
    md("## 2. Phase 14 baseline (frozen reference)"),
    code("import json\n"
         "p14_path = Path('../results/metrics/phase14_readout_diagnosis/summary.json')\n"
         "p14 = json.loads(p14_path.read_text(encoding='utf-8'))\n"
         "base14 = p14['baseline_perf_mean']\n"
         "print('Phase 14 JointCo baseline (original head):')\n"
         "print(f\"  R={base14['R']*100:.1f}%  RC={base14['RC']*100:.1f}%  FRC={base14['FRC']*100:.1f}%  mixed={base14['mixed_mean']*100:.1f}%\")\n"
         "print(f\"  R-signal probe={p14['r_probe_mean']*100:.1f}%  rel_only branch={p14['branch_readout_results_mean']['rel_only']['R']*100:.1f}%\")"),
    md("## 3. Exact current relational computation (15A audit — from the implementation itself)"),
    code("from neuroforge.blocks.joint_co_relational import JointCoRelationalBlock\n"
         "import json as _json\n"
         "audit = JointCoRelationalBlock().describe_relational_computation()\n"
         "print(_json.dumps(audit, indent=1))\n"
         "print()\n"
         "print('WHAT: one ring message-passing round; msgs = A_norm @ Linear(h); rel = tanh(Linear([h;msgs]))')\n"
         "print('WHEN: inside JointCo after the encoder, parallel to feature/context branches, before fusion+residual')\n"
         "print('HOW: neighbour info enters only via the single averaged A @ W(h) step')"),
    md("## 4. Depth ablation (15B: depth 1 = baseline, 2, 3; everything else fixed)"),
    code("import json\n"
         "p15 = json.loads(Path('../results/metrics/phase15_relational_diagnosis/summary.json').read_text(encoding='utf-8'))\n"
         "depth = p15['depth_results_mean']\n"
         "for cond in ('baseline', 'depth2', 'depth3'):\n"
         "    d = depth.get(cond, {})\n"
         "    print(f\"{cond:>9}: \" + '  '.join(f\"{f}={d.get(f, 0)*100:.1f}%\" for f in ('F','R','C','FR','RC','FC','FRC')))\n"
         "print()\n"
         "for r in p15['per_seed_results']:\n"
         "    b = r['baseline_eval']['perf']['R']\n"
         "    print(f\"seed {r['seed']}: baseline R={b*100:.1f}%  depth2 R={r['depth2_eval']['perf']['R']*100:.1f}%  depth3 R={r['depth3_eval']['perf']['R']*100:.1f}%\")"),
    md("## 5. Aggregation diagnosis (15C — runs only if depth does not fully explain; gate recorded in summary)"),
    code("ag = p15.get('aggregation_results_mean', {})\n"
         "print('gate:', p15['gate_info'].get('aggregation_run'), '|', '; '.join(p15['gate_info'].get('reasons', [])))\n"
         "for cond, d in ag.items():\n"
         "    print(f\"{cond:>9}: \" + '  '.join(f\"{f}={d.get(f, 0)*100:.1f}%\" for f in ('R','RC','FRC')))\n"
         "if not ag:\n"
         "    print('NOT TESTED (gated off)')"),
    md("## 6. Update-capacity diagnosis (15D — runs only if depth/aggregation do not explain)"),
    code("cp = p15.get('capacity_results_mean', {})\n"
         "print('gate:', p15['aggregates']['capacity'].get('tested'))\n"
         "for cond, d in cp.items():\n"
         "    print(f\"{cond:>22}: \" + '  '.join(f\"{f}={d.get(f, 0)*100:.1f}%\" for f in ('R','RC','FRC')))\n"
         "if not cp:\n"
         "    print('NOT TESTED (gated off)')"),
    md("## 7. Topology controls (15E — normal vs destructive correspondence permutation)"),
    code("topo = p15['aggregates']['topology']\n"
         "print(f\"baseline destruction drop on R: {topo['baseline_drop_R']*100:+.1f}pp\")\n"
         "print(f\"candidate ({topo['candidate']}) destruction drop on R: {topo['candidate_drop_R']*100:+.1f}pp\")\n"
         "print(f\"dependence delta: {topo['candidate_drop_minus_baseline_drop_R']*100:+.1f}pp (drop alone is not reasoning; needs performance)\")"),
    md("## 8. Causal controls (15F — marker ablation, token permutation, relational destruction)"),
    code("import csv\n"
         "rows = list(csv.DictReader(open('../results/metrics/phase15_relational_diagnosis/causal_controls.csv')))\n"
         "cand = p15['candidate_cond']\n"
         "for row in rows:\n"
         "    if row['seed'] == str(p15['per_seed_results'][0]['seed']) and row['condition'] == cand:\n"
         "        print(f\"{row['control']:>11} {row['variant']:>22}: R={float(row['R'])*100:.1f}% RC={float(row['RC'])*100:.1f}% FRC={float(row['FRC'])*100:.1f}%\")"),
    md("## 9. Branch ablations (15G — F/R/C only and pairs for the best candidate)"),
    code("cand = p15['candidate_cond']\n"
         "br = p15['per_seed_results'][0][cand + '_eval']['branch7']\n"
         "for combo, perf in br.items():\n"
         "    print(f\"{combo:>6}: \" + '  '.join(f\"{f}={perf.get(f, 0)*100:.1f}%\" for f in ('R','RC','FRC')))\n"
         "print()\n"
         "print('Required check: if R improves but RC/FRC do not -> isolated capability, no compositional transfer.')"),
    md("## 10. Representation probe reassessment (15H — baseline vs candidate)"),
    code("g = p15['aggregates']['gap']\n"
         "print(f\"baseline:  probe R={g['baseline_probe_R_mean']*100:.1f}%  final R={g['baseline_final_R_mean']*100:.1f}%  gap={g['baseline_gap_mean']*100:.1f}pp\")\n"
         "print(f\"candidate: probe R={g['candidate_probe_R_mean']*100:.1f}%  final R={g['candidate_final_R_mean']*100:.1f}%  gap={g['candidate_gap_mean']*100:.1f}pp\")\n"
         "print(f\"gap closes {g['gap_close_mean']*100:+.1f}pp\")"),
    md("## 11. Standalone Graph comparison (15I — is JointCo's relational path weaker than Graph?)"),
    code("import csv\n"
         "rows = list(csv.DictReader(open('../results/metrics/phase15_relational_diagnosis/graph_comparison.csv')))\n"
         "seen = set()\n"
         "for row in rows:\n"
         "    key = (row['seed'], row['condition'])\n"
         "    if row['seed'] == str(p15['per_seed_results'][0]['seed']) and key not in seen:\n"
         "        seen.add(key)\n"
         "        print(f\"{row['condition']:>18}: R={float(row['R'])*100:.1f}% drop={float(row['destruction_drop_R'])*100:+.1f}pp probe={float(row['probe_R'])*100:.1f}% params={row['params']}\")"),
    md("## 12. Compute and latency (FLOPs are not latency — both reported)"),
    code("import csv\n"
         "for name in ('compute.csv', 'latency.csv'):\n"
         "    print(f'--- {name} (seed {p15[\"per_seed_results\"][0][\"seed\"]}) ---')\n"
         "    for row in csv.DictReader(open(f'../results/metrics/phase15_relational_diagnosis/{name}')):\n"
         "        if row['seed'] == str(p15['per_seed_results'][0]['seed']):\n"
         "            print(' ', {k: v for k, v in row.items() if k != 'seed'})"),
    md("## 13. Seed aggregation (mean ± SD, never best-seed)"),
    code("import csv\n"
         "for row in csv.DictReader(open('../results/metrics/phase15_relational_diagnosis/seed_results.csv')):\n"
         "    print(' ', {k: (f'{float(v)*100:.1f}%' if k != 'seed' else v) for k, v in row.items()})\n"
         "print('stability (R seed-SD):', {k: {c: f'{v*100:.1f}pp' for c, v in d.items()} for k, d in p15['aggregates']['seed_stability'].items()})"),
    md("## 14. H1–H8 verdicts (re-derived programmatically from the aggregates)"),
    code("from neuroforge.evaluation.phase15_metrics import build_phase15_hypotheses\n"
         "hyps = build_phase15_hypotheses(p15['aggregates'])\n"
         "for h in sorted(hyps):\n"
         "    print(f\"{h}: {hyps[h]['status']}\")\n"
         "    print(f\"    {hyps[h]['evidence']}\")\n"
         "assert set(hyps) == set(p15['hypotheses'])\n"
         "assert all(hyps[h]['status'] == p15['hypotheses'][h]['status'] for h in hyps)\n"
         "print('stored verdicts match fresh derivation: OK')"),
    md("## 15. Final CASE classification (programmatic — no manually typed numbers)"),
    code("from neuroforge.evaluation.phase15_metrics import select_phase15_case\n"
         "case, label = select_phase15_case(hyps, p15['aggregates']['gap']['baseline_gap_mean'],\n"
         "                                 p15['aggregates']['compositional'].get('R_gain_mean', 0.0))\n"
         "print(f'Programmatic verdict: {case} — {label}')\n"
         "assert case == p15['verdict_case'], 'stored case must equal fresh derivation'\n"
         "print(f\"Minimal validated intervention: {p15['minimal_intervention']['intervention']} ({p15['minimal_intervention']['outcome']})\")\n"
         "print()\n"
         "print('Summary of evidence (all values loaded, none typed):')\n"
         "print(f\"  - baseline R: {p15['baseline_perf_mean']['R']*100:.1f}%, probe gap: {p15['aggregates']['gap']['baseline_gap_mean']*100:.1f}pp\")\n"
         "print(f\"  - H1: {hyps['H1']['status']}, H2: {hyps['H2']['status']}, H3: {hyps['H3']['status']}, H4: {hyps['H4']['status']}\")\n"
         "print(f\"  - H5: {hyps['H5']['status']}, H6: {hyps['H6']['status']}, H7: {hyps['H7']['status']}, H8: {hyps['H8']['status']}\")"),
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

out = Path("notebooks/23_relational_substep_diagnosis.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
