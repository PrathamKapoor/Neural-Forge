"""CLI runner for Phase 15 relational-substep diagnosis (staged, resumable)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroforge.training.phase15_relational_diagnosis import (
    generate_phase15_markdown_report,
    run_phase15_relational_diagnosis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 Relational Substep Diagnosis")
    parser.add_argument("--output-dir", type=str, default="results/phase15_relational_diagnosis")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/phase15_relational_diagnosis")
    parser.add_argument("--figures-dir", type=str, default="figures/phase15_relational_diagnosis")
    parser.add_argument("--report-path", type=str, default="results/reports/phase15_relational_diagnosis.md")
    parser.add_argument("--jointco-epochs", type=int, default=40)
    parser.add_argument("--samples-per-type", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument(
        "--stages", type=str, nargs="+",
        default=["baseline", "depth", "gated", "graph", "existing", "finalize"],
    )
    args = parser.parse_args()

    print("=" * 72)
    print("NeuroForge Phase 15: Relational Substep Capability Diagnosis")
    print("=" * 72)

    summary = run_phase15_relational_diagnosis(
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        seeds=tuple(args.seeds),
        jointco_epochs=args.jointco_epochs,
        samples_per_type=args.samples_per_type,
        batch_size=args.batch_size,
        stages=tuple(args.stages),
    )

    if "verdict_case" in summary:
        report_path = Path(args.report_path)
        generate_phase15_markdown_report(summary, report_path)
        print(f"Report written to: {report_path}")
        print(f"  - Verdict: {summary.get('verdict_case', '?')} - {summary.get('verdict_label', '')}")
        print(f"  - Intervention: {summary.get('minimal_intervention', {}).get('intervention', '?')}")
    else:
        print(f"Intermediate stages complete; gate info: {summary.get('gate_info', {})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
