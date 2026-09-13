"""CLI runner for Phase 9 Generalization and Mixed-Structure Evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroforge.training.phase9_generalization import (
    generate_markdown_report,
    run_phase9_generalization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 Generalization and Mixed-Structure Evaluation")
    parser.add_argument("--output-dir", type=str, default="results/phase9_generalization")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/phase9_generalization")
    parser.add_argument("--figures-dir", type=str, default="figures/phase9_generalization")
    parser.add_argument("--report-path", type=str, default="results/reports/phase9_generalization.md")
    parser.add_argument("--expert-epochs", type=int, default=25)
    parser.add_argument("--router-epochs", type=int, default=30)
    args = parser.parse_args()

    print("=" * 72)
    print("NeuroForge Phase 9: Generalization & Mixed-Structure Evaluation")
    print("=" * 72)

    summary = run_phase9_generalization(
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        expert_epochs=args.expert_epochs,
        router_epochs=args.router_epochs,
    )

    report_path = Path(args.report_path)
    generate_markdown_report(summary, report_path)
    print(f"Report written to: {report_path}")

    # Executive Summary Console Output
    print("\nPhase 9 Completed Successfully:")
    print(f"  - Summary JSON: {args.metrics_dir}/phase9_summary.json")
    print(f"  - Figures Generated: {len(summary.get('generated_figures', []))} figures in {args.figures_dir}")
    print(f"  - Comprehensive Report: {args.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
