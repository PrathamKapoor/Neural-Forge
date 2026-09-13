"""CLI runner for Phase 16 embedded-vs-dedicated relational diagnosis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroforge.training.phase16_relational_interface_diagnosis import (
    generate_phase16_markdown_report,
    run_phase16_relational_interface,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 16 Relational Interface Diagnosis")
    parser.add_argument("--output-dir", type=str, default="results/phase16_relational_interface")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/phase16_relational_interface")
    parser.add_argument("--figures-dir", type=str, default="figures/phase16_relational_interface")
    parser.add_argument("--report-path", type=str, default="results/reports/phase16_relational_interface.md")
    parser.add_argument("--jointco-epochs", type=int, default=40)
    parser.add_argument("--samples-per-type", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    args = parser.parse_args()

    print("=" * 72)
    print("NeuroForge Phase 16: Embedded vs Dedicated Relational Diagnosis")
    print("=" * 72)

    summary = run_phase16_relational_interface(
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        seeds=tuple(args.seeds),
        jointco_epochs=args.jointco_epochs,
        samples_per_type=args.samples_per_type,
        batch_size=args.batch_size,
    )

    report_path = Path(args.report_path)
    generate_phase16_markdown_report(summary, report_path)
    print(f"Report written to: {report_path}")
    print(f"  - Verdict: {summary.get('verdict_case', '?')} - {summary.get('verdict_label', '')}")
    print(f"  - Intervention: {summary.get('minimal_intervention', {}).get('intervention', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
