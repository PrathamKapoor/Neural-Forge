"""CLI runner for Phase 13 Joint Co-Adaptation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroforge.training.phase13_joint_coadaptation import (
    generate_phase13_markdown_report,
    run_phase13_joint_coadaptation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 13 Joint Co-Adaptation")
    parser.add_argument("--output-dir", type=str, default="results/phase13_joint_coadaptation")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/phase13_joint_coadaptation")
    parser.add_argument("--figures-dir", type=str, default="figures/phase13_joint_coadaptation")
    parser.add_argument("--report-path", type=str, default="results/reports/phase13_joint_coadaptation.md")
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--extended-epochs", type=int, default=40)
    parser.add_argument("--samples-per-type", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=60)
    args = parser.parse_args()

    print("=" * 72)
    print("NeuroForge Phase 13: Joint Co-Adaptation")
    print("=" * 72)

    summary = run_phase13_joint_coadaptation(
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        baseline_epochs=args.baseline_epochs,
        extended_epochs=args.extended_epochs,
        samples_per_type=args.samples_per_type,
        batch_size=args.batch_size,
    )

    report_path = Path(args.report_path)
    generate_phase13_markdown_report(summary, report_path)
    print(f"Report written to: {report_path}")

    print("\nPhase 13 Completed:")
    print(f"  - Metrics & Manifest: {args.metrics_dir}/")
    print(f"  - Figures: {len(summary.get('generated_figures', []))} in {args.figures_dir}")
    print(f"  - Report: {args.report_path}")
    print(f"  - Verdict: {summary.get('verdict_case', '?')} - {summary.get('verdict_label', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
