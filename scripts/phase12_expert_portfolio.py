"""CLI runner for Phase 12 Expert Portfolio Expansion."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroforge.training.phase12_expert_portfolio import (
    generate_phase12_markdown_report,
    run_phase12_expert_portfolio,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 12 Expert Portfolio Expansion")
    parser.add_argument("--output-dir", type=str, default="results/phase12_expert_portfolio")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/phase12_expert_portfolio")
    parser.add_argument("--figures-dir", type=str, default="figures/phase12_expert_portfolio")
    parser.add_argument("--report-path", type=str, default="results/reports/phase12_expert_portfolio.md")
    parser.add_argument("--expert-epochs", type=int, default=20)
    parser.add_argument("--samples-per-type", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=60)
    args = parser.parse_args()

    print("=" * 72)
    print("NeuroForge Phase 12: Expert Portfolio Expansion")
    print("=" * 72)

    summary = run_phase12_expert_portfolio(
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        expert_epochs=args.expert_epochs,
        samples_per_type=args.samples_per_type,
        batch_size=args.batch_size,
    )

    report_path = Path(args.report_path)
    generate_phase12_markdown_report(summary, report_path)
    print(f"Report written to: {report_path}")

    print("\nPhase 12 Completed:")
    print(f"  - Metrics & Manifest: {args.metrics_dir}/")
    print(f"  - Figures: {len(summary.get('generated_figures', []))} in {args.figures_dir}")
    print(f"  - Report: {args.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
