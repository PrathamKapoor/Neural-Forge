"""CLI runner for Phase 10 Adaptive Multi-Expert Composition."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroforge.training.phase10_multi_expert import (
    generate_phase10_markdown_report,
    run_phase10_multi_expert,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 10 Adaptive Multi-Expert Composition")
    parser.add_argument("--output-dir", type=str, default="results/phase10_multi_expert")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/phase10_multi_expert")
    parser.add_argument("--figures-dir", type=str, default="figures/phase10_multi_expert")
    parser.add_argument("--report-path", type=str, default="results/reports/phase10_multi_expert.md")
    parser.add_argument("--expert-epochs", type=int, default=25)
    parser.add_argument("--router-epochs", type=int, default=25)
    parser.add_argument("--samples-per-type", type=int, default=120)
    args = parser.parse_args()

    print("=" * 72)
    print("NeuroForge Phase 10: Adaptive Multi-Expert Composition")
    print("=" * 72)

    summary = run_phase10_multi_expert(
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        expert_epochs=args.expert_epochs,
        router_epochs=args.router_epochs,
        samples_per_type=args.samples_per_type,
    )

    report_path = Path(args.report_path)
    generate_phase10_markdown_report(summary, report_path)
    print(f"Report written to: {report_path}")

    print("\nPhase 10 Completed Successfully:")
    print(f"  - Metrics & Manifest: {args.metrics_dir}/")
    print(f"  - Figures Generated: {len(summary.get('generated_figures', []))} figures in {args.figures_dir}")
    print(f"  - Comprehensive Report: {args.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
