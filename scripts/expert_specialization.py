"""Run Phase 3 and write source metrics and an evidence-only report."""
from pathlib import Path
import json
from neuroforge.training import run_expert_specialization

out = Path("results/metrics/phase3_expert_specialization")
result = run_expert_specialization(out)
Path("results/reports").mkdir(parents=True, exist_ok=True)
lines = ["# Phase 3 — Capacity-matched expert specialization", "", "## Results", "", "| Architecture | Feature | Relational | Contextual |", "|---|---:|---:|---:|"]
for architecture, values in result["matrix"].items(): lines.append(f"| {architecture} | " + " | ".join(f"{x:.3f}" for x in values) + " |")
lines += ["", "## Capacity and controls", "", f"Depths: {result['capacity']['depths']}; parameters: {result['capacity']['parameter_counts']}; maximum relative gap: {result['capacity']['max_relative_gap']:.1%}.", "", "GraphBlock constructs its ring internally, so literal external edge randomization was not implemented. The relational control is deterministic node-feature permutation relative to fixed ring topology: it retains values, shape, degree, parameters, and block implementation while disrupting feature/topology correspondence.", "", "## Interpretation", "", "These serialized data are the authoritative evidence. Rankings and controls must be assessed together; no claim of learned routing is made in Phase 3."]
Path("results/reports/phase3_expert_specialization.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(result["matrix"], indent=2))
